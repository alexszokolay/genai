# worker.py
import os
import tempfile
from io import BytesIO

import whisper
from openai import OpenAI

# ---------- OpenAI client (for Chat + TTS) ----------
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")
client = OpenAI(api_key=API_KEY)

# ---------- Whisper model (local STT) ----------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")   # tiny|base|small|medium|large
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" or "cuda"
_whisper = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)

def speech_to_text(audio_binary: bytes, language: str = "en") -> str:
    """Local STT via Whisper (no API usage). Requires ffmpeg installed."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_binary)
        tmp.flush()
        result = _whisper.transcribe(tmp.name, language=language, fp16=False)
    return (result.get("text") or "").strip()

def text_to_speech(text: str, voice: str = "alloy", fmt: str = "mp3") -> bytes:
    """
    OpenAI TTS -> audio bytes. If you're out of API quota, either:
      - switch to a local TTS, or
      - temporarily skip TTS and only return text.
    """
    resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice or "alloy",
        input=text,
        format=fmt,     # "mp3" | "wav" | "opus" | "pcm16"
    )
    return resp.read()

def openai_process_message(user_message: str) -> str:
    """Chat completion using a current small 4o-family model."""
    system_prompt = (
        "Act like a helpful personal assistant. Be concise and practical. "
        "You can answer questions, translate, summarize, and recommend."
    )
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=600,
    )
    return chat.choices[0].message.content.strip()

