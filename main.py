from app.server import app
from app.server import ensure_llama_tts
from app.config import CONFIG

if __name__ == '__main__':
    import uvicorn
    ensure_llama_tts()
    uvicorn.run(
        app,
        host=CONFIG["app"]["host"],
        port=CONFIG["app"]["port"],
        log_config=None,
        access_log=True,
    )
