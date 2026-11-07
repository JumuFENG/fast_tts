import base64
import socket
import subprocess
import time
from pathlib import Path

import httpx2 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi import UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
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
VOICE_DIR = Path(CONFIG["voices"]["directory"])
VOICES = CONFIG["voices"]["items"]
DEFAULT_VOICE = CONFIG["voices"]["default"]


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
    text: str = Field(min_length=1, max_length=2000)
    voice: str = DEFAULT_VOICE
    lang: str = "zh"


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



@app.post("/tts")
async def generate_tts(req: TTSRequest):
    voice = VOICES.get(req.voice)
    if voice is None:
        raise HTTPException(400, f"Unknown voice: {req.voice}")

    speaker_file = VOICE_DIR / voice["file"]

    payload = {
        "input": req.text,
        "lang": req.lang,
        "stream": True,
        "response_format": "wav",
        "n_predict": MAX_AUDIO_FRAMES,
    }

    if speaker_file.is_file():
        payload["speaker_ref_b64"] = base64.b64encode(
            speaker_file.read_bytes()
        ).decode("ascii")

    client = httpx2.AsyncClient(timeout=300)

    try:
        request = client.build_request(
            "POST",
            f"{LLAMA_URL}/tts",
            json=payload,
        )

        # 开启流式请求
        upstream = await client.send(request, stream=True)

        # 检查上游响应
        if upstream.status_code >= 400:
            error = await upstream.aread()
            status_code = upstream.status_code

            await upstream.aclose()
            await client.aclose()

            raise HTTPException(
                status_code,
                error.decode("utf-8", errors="replace"),
            )

    except HTTPException:
        raise
    except httpx2.RequestError as e:
        await client.aclose()
        raise HTTPException(
            502,
            f"Cannot connect to llama-server: {e}",
        )

    # 逐块转发音频
    async def audio_stream():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
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



@app.post("/tts/upload")
async def generate_tts_upload(
    text: str = Form(...),
    speaker_ref: UploadFile = File(...),
    lang: str = Form("zh"),
):
    if speaker_ref.content_type not in ("audio/wav", "audio/mpeg"):
        raise HTTPException(400, "Only WAV or MP3 is supported")

    audio = await speaker_ref.read()
    await speaker_ref.close()

    if not audio or len(audio) > 10 * 1024 * 1024:
        raise HTTPException(400, "Audio must be between 1 byte and 10 MB")

    client = httpx2.AsyncClient(timeout=300)

    try:
        request = client.build_request(
            "POST",
            f"{LLAMA_URL}/tts",
            json={
                "input": text,
                "lang": lang,
                "stream": True,
                "response_format": "wav",
                "n_predict": MAX_AUDIO_FRAMES,
                "speaker_ref_b64": base64.b64encode(audio).decode("ascii"),
            },
        )

        upstream = await client.send(request, stream=True)

        if upstream.status_code >= 400:
            error = await upstream.aread()
            status_code = upstream.status_code

            await upstream.aclose()
            await client.aclose()

            raise HTTPException(
                status_code,
                error.decode("utf-8", errors="replace"),
            )

    except HTTPException:
        raise
    except httpx2.RequestError as e:
        await client.aclose()
        raise HTTPException(
            502,
            f"Cannot connect to llama-server: {e}",
        )

    async def audio_stream():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
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
