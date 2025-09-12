# worker.py (top of file)
import os
import tempfile
from io import BytesIO

import whisper
from openai import OpenAI

# Load Whisper model once at import
# Options: tiny, base, small, medium, large  (bigger = slower but more accurate)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = "cpu"  # set "cuda" if you have a GPU configured
_whisper_model = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)

openai_client = OpenAI()  # unchanged (for chat + TTS)

def speech_to_text(audio_binary: bytes, language: str = "en") -> str:
    """
    Local STT using OpenAI Whisper (runs on your machine).
    Requires ffmpeg installed on the OS.
    Accepts raw bytes (wav/mp3/m4a/webm/ogg/flac, etc.).
    """
    # Write bytes to a temp file so Whisper/ffmpeg can read it
    # Using .webm suffix is fine; ffmpeg detects real format from contents.
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
        tmp.write(audio_binary)
        tmp.flush()
        # fp16=False is important on CPU; set True only for CUDA
        result = _whisper_model.transcribe(
            tmp.name,
            language=language,
            fp16=False
        )
    return (result.get("text") or "").strip()
