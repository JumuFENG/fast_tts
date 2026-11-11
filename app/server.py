import base64
import asyncio
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Literal

import httpx2 
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from app.config import CONFIG

app = FastAPI(title="Local Qwen TTS")


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

LLAMA_HOST = CONFIG["llama_tts"]["host"]
LLAMA_PORT = CONFIG["llama_tts"]["port"]
LLAMA_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
LLAMA_COMMAND = CONFIG["llama_tts"]["command"]
STARTUP_TIMEOUT = CONFIG["llama_tts"]["startup_timeout"]
MAX_AUDIO_FRAMES = CONFIG["llama_tts"]["max_audio_frames"]
MAX_TEXT_LENGTH = CONFIG["llama_tts"]["max_text_length"]
VOICE_DIR = Path(CONFIG["voices"]["directory"])
VOICES = CONFIG["voices"]["items"]
DEFAULT_VOICE = CONFIG["voices"]["default"]
FFMPEG = CONFIG["app"]["ffmpeg"]


def ensure_llama_tts():
    def is_running():
        try:
            with socket.create_connection((LLAMA_HOST, LLAMA_PORT), timeout=1):
                return True
        except OSError:
            return False

    if is_running():
        return
    if not LLAMA_COMMAND:
        raise RuntimeError("llama-tts is not running and no startup command is configured")

    command = [arg.format(host=LLAMA_HOST, port=LLAMA_PORT) for arg in LLAMA_COMMAND]
    process = subprocess.Popen(command)
    process_name = Path(command[0]).name
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if is_running():
            return
        if process.poll() is not None:
            raise RuntimeError(
                f"{process_name} exited during startup (exit code {process.returncode})"
            )
        time.sleep(0.5)
    raise RuntimeError(
        f"{process_name} did not listen on {LLAMA_HOST}:{LLAMA_PORT} "
        f"within {STARTUP_TIMEOUT} seconds"
    )


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str = DEFAULT_VOICE
    lang: str = "zh"
    response_format: Literal["mp3", "wav"] = "mp3"


class OpenAISpeechRequest(BaseModel):
    model: str
    input: str = Field(min_length=1, max_length=4096)
    voice: str = "alloy"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = None
    stream_format: Literal["audio", "sse"] = "audio"


OPENAI_VOICES = CONFIG.get("openai_tts", {}).get("voices", {})


@app.get("/voices")
async def get_voices():
    return {
        "default": DEFAULT_VOICE,
        "voices": [
            {
                "id": name,
                "description": info["description"],
            }
            for name, info in VOICES.items()
            if (VOICE_DIR / info["file"]).is_file() or name == DEFAULT_VOICE
        ],
    }


def get_migpt_voice(speaker):
    if not speaker:
        return VOICES[DEFAULT_VOICE]
    for voice_id, voice in VOICES.items():
        if voice.get("speaker") == speaker:
            if voice_id != DEFAULT_VOICE and not (VOICE_DIR / voice["file"]).is_file():
                raise HTTPException(400, f"Speaker is unavailable: {speaker}")
            return voice
    raise HTTPException(400, f"Unknown speaker: {speaker}")


@app.get("/api/speakers")
async def get_migpt_speakers():
    return [
        {
            "name": voice["name"],
            "gender": voice["gender"],
            "speaker": voice["speaker"],
        }
        for voice_id, voice in VOICES.items()
        if (VOICE_DIR / voice["file"]).is_file() or voice_id == DEFAULT_VOICE
    ]


@app.get("/api/tts.mp3")
async def generate_migpt_tts(
    text: str = Query(..., min_length=1),
    speaker: str | None = None,
):
    if not text.strip():
        raise HTTPException(422, "Text must not be blank")
    voice = get_migpt_voice(speaker)
    voice_file = VOICE_DIR / voice["file"]
    speaker_ref_b64 = None
    if voice_file.is_file():
        speaker_ref_b64 = base64.b64encode(voice_file.read_bytes()).decode("ascii")

    return await generate_tts_response(text, "zh", speaker_ref_b64, "mp3")



def split_text(text):
    remaining = text.strip()
    segments = []
    while len(remaining) > MAX_TEXT_LENGTH:
        end = MAX_TEXT_LENGTH
        for index in range(end - 1, 0, -1):
            if remaining[index] in "。！？.!?\n":
                end = index + 1
                break
        segments.append(remaining[:end])
        remaining = remaining[end:]
    if remaining:
        segments.append(remaining)
    return segments


def wav_data_offset(header):
    if len(header) < 12:
        return None
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise RuntimeError("llama-server returned a non-WAV response")

    offset = 12
    while offset + 8 <= len(header):
        chunk_id = header[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", header, offset + 4)[0]
        if chunk_id == b"data":
            return offset + 8, offset + 4
        if chunk_size > 1024 * 1024:
            raise RuntimeError("WAV header contains an unexpectedly large chunk")
        next_offset = offset + 8 + chunk_size + (chunk_size & 1)
        if next_offset > len(header):
            return None
        offset = next_offset
    return None


async def start_upstream(client, text, lang, speaker_ref_b64=None, stream_started=False):
    payload = {
        "input": text,
        "lang": lang,
        "stream": True,
        "response_format": "wav",
        "n_predict": MAX_AUDIO_FRAMES,
    }
    if speaker_ref_b64:
        payload["speaker_ref_b64"] = speaker_ref_b64

    try:
        request = client.build_request("POST", f"{LLAMA_URL}/tts", json=payload)
        upstream = await client.send(request, stream=True)
    except httpx2.RequestError as error:
        if stream_started:
            raise RuntimeError(f"Cannot continue llama-server TTS stream: {error}") from error
        raise HTTPException(502, f"Cannot connect to llama-server: {error}") from error

    if upstream.status_code >= 400:
        error = (await upstream.aread()).decode("utf-8", errors="replace")
        status_code = upstream.status_code
        await upstream.aclose()
        if stream_started:
            raise RuntimeError(f"llama-server failed on a later text segment ({status_code}): {error}")
        raise HTTPException(status_code, error)
    return upstream


async def generate_tts_stream(text, lang, speaker_ref_b64=None):
    segments = split_text(text)
    if not segments:
        raise HTTPException(422, "Text must not be blank")

    client = httpx2.AsyncClient(timeout=300)
    try:
        first_upstream = await start_upstream(client, segments[0], lang, speaker_ref_b64)
    except Exception:
        await client.aclose()
        raise

    async def audio_stream():
        upstream = first_upstream
        template_header = None
        try:
            for index, segment in enumerate(segments):
                if index:
                    upstream = await start_upstream(
                        client, segment, lang, speaker_ref_b64, stream_started=True
                    )

                buffered = bytearray()
                header_found = False
                async for chunk in upstream.aiter_bytes():
                    if header_found:
                        yield chunk
                        continue

                    buffered.extend(chunk)
                    parsed = wav_data_offset(buffered)
                    if parsed is None:
                        continue

                    data_offset, data_size_offset = parsed
                    header = bytearray(buffered[:data_offset])
                    header_key = bytearray(header)
                    struct.pack_into("<I", header_key, 4, 0)
                    struct.pack_into("<I", header_key, data_size_offset, 0)
                    if index == 0:
                        template_header = header_key
                        struct.pack_into("<I", header, 4, 0xFFFFFFFF)
                        struct.pack_into("<I", header, data_size_offset, 0xFFFFFFFF)
                        yield bytes(header)
                    elif header_key != template_header:
                        raise RuntimeError("WAV format changed between text segments")

                    audio_data = buffered[data_offset:]
                    if audio_data:
                        yield bytes(audio_data)
                    header_found = True
                    buffered.clear()

                if not header_found:
                    raise RuntimeError("llama-server returned an incomplete WAV header")
                await upstream.aclose()
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="speech.wav"',
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def transcode_audio_stream(wav_response, response_format="mp3", speed=1.0):
    output_formats = {
        "mp3": ("libmp3lame", "mp3", "audio/mpeg", "speech.mp3"),
        "opus": ("libopus", "opus", "audio/ogg", "speech.opus"),
        "aac": ("aac", "adts", "audio/aac", "speech.aac"),
        "flac": ("flac", "flac", "audio/flac", "speech.flac"),
        "wav": ("pcm_s16le", "wav", "audio/wav", "speech.wav"),
        "pcm": ("pcm_s16le", "s16le", "audio/pcm", "speech.pcm"),
    }
    codec, container, media_type, filename = output_formats[response_format]
    atempo = []
    tempo = speed
    while tempo < 0.5:
        atempo.append("atempo=0.5")
        tempo *= 2
    while tempo > 2.0:
        atempo.append("atempo=2.0")
        tempo /= 2
    if speed != 1.0:
        atempo.append(f"atempo={tempo:g}")
    try:
        command = [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vn",
        ]
        if atempo:
            command.extend(["-filter:a", ",".join(atempo)])
        command.extend(["-codec:a", codec])
        if response_format == "mp3":
            command.extend(["-q:a", "4"])
        command.extend(["-flush_packets", "1", "-f", container, "pipe:1"])
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as error:
        await wav_response.body_iterator.aclose()
        raise HTTPException(503, f"Cannot start FFmpeg ({FFMPEG}): {error}") from error

    async def feed_ffmpeg():
        try:
            async for chunk in wav_response.body_iterator:
                process.stdin.write(chunk)
                await process.stdin.drain()
        finally:
            process.stdin.close()
            await process.stdin.wait_closed()

    async def read_errors():
        return await process.stderr.read()

    async def mp3_stream():
        feeder = asyncio.create_task(feed_ffmpeg())
        errors = asyncio.create_task(read_errors())
        try:
            while chunk := await process.stdout.read(64 * 1024):
                yield chunk
            await feeder
            error_output = await errors
            return_code = await process.wait()
            if return_code:
                message = error_output.decode("utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg audio conversion failed: {message}")
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            for task in (feeder, errors):
                if not task.done():
                    task.cancel()
            await asyncio.gather(feeder, errors, return_exceptions=True)

    return StreamingResponse(
        mp3_stream(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def generate_mp3_stream(wav_response):
    return await transcode_audio_stream(wav_response)


async def generate_tts_response(text, lang, speaker_ref_b64, response_format):
    if response_format == "wav":
        return await generate_tts_stream(text, lang, speaker_ref_b64)
    if response_format != "mp3":
        raise HTTPException(400, "response_format must be 'mp3' or 'wav'")
    wav_response = await generate_tts_stream(text, lang, speaker_ref_b64)
    return await generate_mp3_stream(wav_response)


@app.post("/v1/audio/speech")
async def openai_create_speech(req: OpenAISpeechRequest):
    if not req.input.strip():
        raise HTTPException(422, "input must not be blank")
    if req.stream_format == "sse":
        raise HTTPException(400, "stream_format='sse' is not supported; use 'audio'")
    if req.instructions:
        raise HTTPException(400, "instructions are not supported by the configured llama-tts model")

    voice_id = OPENAI_VOICES.get(req.voice, req.voice)
    voice = VOICES.get(voice_id)
    if voice is None:
        raise HTTPException(400, f"Unknown voice: {req.voice}")
    speaker_file = VOICE_DIR / voice["file"]
    if voice_id != DEFAULT_VOICE and not speaker_file.is_file():
        raise HTTPException(400, f"Voice is unavailable: {req.voice}")
    speaker_ref_b64 = None
    if speaker_file.is_file():
        speaker_ref_b64 = base64.b64encode(speaker_file.read_bytes()).decode("ascii")

    wav_response = await generate_tts_stream(req.input, "zh", speaker_ref_b64)
    if req.response_format == "wav" and req.speed == 1.0:
        return wav_response
    return await transcode_audio_stream(wav_response, req.response_format, req.speed)


@app.post("/tts")
async def generate_tts(req: TTSRequest):
    voice = VOICES.get(req.voice)
    if voice is None:
        raise HTTPException(400, f"Unknown voice: {req.voice}")

    speaker_file = VOICE_DIR / voice["file"]
    speaker_ref_b64 = None
    if speaker_file.is_file():
        speaker_ref_b64 = base64.b64encode(
            speaker_file.read_bytes()
        ).decode("ascii")
    return await generate_tts_response(
        req.text, req.lang, speaker_ref_b64, req.response_format
    )



@app.post("/tts/upload")
async def generate_tts_upload(
    text: str = Form(...),
    speaker_ref: UploadFile = File(...),
    lang: str = Form("zh"),
    response_format: Literal["mp3", "wav"] = Form("mp3"),
):
    if speaker_ref.content_type not in ("audio/wav", "audio/mpeg"):
        raise HTTPException(400, "Only WAV or MP3 is supported")

    audio = await speaker_ref.read()
    await speaker_ref.close()

    if not audio or len(audio) > 10 * 1024 * 1024:
        raise HTTPException(400, "Audio must be between 1 byte and 10 MB")
    if not text.strip():
        raise HTTPException(422, "Text must not be blank")

    speaker_ref_b64 = base64.b64encode(audio).decode("ascii")
    return await generate_tts_response(
        text, lang, speaker_ref_b64, response_format
    )
