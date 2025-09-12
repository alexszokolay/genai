# server.py
import base64
import json
import os
from flask import Flask, render_template, request, Response, jsonify
from flask_cors import CORS
from worker import speech_to_text, text_to_speech, openai_process_message

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/speech-to-text", methods=["POST", "OPTIONS"])
def speech_to_text_route():
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return _cors_ok()

    try:
        print("processing /speech-to-text")

        # Accept audio as multipart file "audio" OR raw body
        audio_bytes = None
        if "audio" in request.files:
            audio_bytes = request.files["audio"].read()
        else:
            # raw body
            audio_bytes = request.get_data(cache=False)

        if not audio_bytes:
            return jsonify({"error": "No audio provided"}), 400

        text = speech_to_text(audio_bytes)
        return jsonify({"text": text or ""}), 200

    except Exception as e:
        print("speech-to-text error:", repr(e))
        return jsonify({"error": str(e)}), 500

@app.route("/process-message", methods=["POST", "OPTIONS"])
def process_message_route():
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return _cors_ok()

    try:
        payload = request.get_json(silent=True) or {}
        user_message = (payload.get("userMessage") or "").strip()
        voice = (payload.get("voice") or "alloy").strip()  # default to alloy for OpenAI TTS

        if not user_message:
            return jsonify({"error": "Missing 'userMessage'"}), 400

        # LLM reply
        openai_response_text = openai_process_message(user_message)
        # Clean blank lines
        openai_response_text = os.linesep.join([s for s in openai_response_text.splitlines() if s])

        # TTS bytes -> base64 string (mp3 by default in worker.text_to_speech)
        audio_bytes = text_to_speech(openai_response_text, voice)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        resp = {
            "openaiResponseText": openai_response_text,
            "openaiResponseSpeech": audio_b64,
            # Optional: tell the front-end what MIME to expect if it wants to play inline
            "audioMimeType": "audio/mpeg"
        }
        return app.response_class(
            response=json.dumps(resp),
            status=200,
            mimetype="application/json"
        )
    except Exception as e:
        print("process-message error:", repr(e))
        return jsonify({"error": str(e)}), 500

def _cors_ok():
    # Small helper for preflight responses if your frontend sends OPTIONS
    r = Response()
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return r

if __name__ == "__main__":
    # If you want auto-reload in dev: app.run(port=8000, host="0.0.0.0", debug=True)
    app.run(port=8000, host="0.0.0.0")
