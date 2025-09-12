# worker.py
from io import BytesIO
from openai import OpenAI

client = OpenAI()

def speech_to_text(audio_binary: bytes) -> str:
    """
    Convert raw audio bytes to text using OpenAI STT.
    Accepts WAV/MP3/M4A/WEBM/OGG/FLAC. If your recorder always uses WAV, that's fine.
    """
    # Prefer gpt-4o-mini-transcribe for speed/quality; fallback: "whisper-1"
    model = "gpt-4o-mini-transcribe"
    file_obj = BytesIO(audio_binary)
    file_obj.name = "audio.wav"        # a filename hint helps the SDK pick a mime type

    result = client.audio.transcriptions.create(
        model=model,
        file=file_obj,
        # optional: temperature=0, language="en"
    )
    # SDK returns an object with .text
    return (result.text or "").strip()

def text_to_speech(text: str, voice: str = "alloy") -> bytes:
    """
    Convert text to speech using OpenAI TTS. Returns MP3 bytes.
    Voices commonly available: alloy, verse, aria, coral, sage, breeze (varies by account).
    """
    # gpt-4o-mini-tts supports multiple voices; format can be "mp3" | "wav" | "opus" | "pcm16"
    resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice or "alloy",
        input=text,
        format="mp3"
    )
    # In the Python SDK v1, .audio.speech.create returns a BinaryResponse
    # Use .read() to get raw bytes
    return resp.read()

def openai_process_message(user_message: str) -> str:
    """
    Lightweight assistant behavior using a current small 4o family model.
    """
    system_prompt = (
        "Act like a helpful personal assistant. You can respond to questions, "
        "translate sentences, summarize news, and give recommendations. "
        "Be concise and practical."
    )

    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=600
    )
    return chat.choices[0].message.content.strip()
