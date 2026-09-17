FROM python:3.11-slim

# ffmpeg: safety net for pydub's audio decoding (STT path tries an explicit
# WAV decode first and only falls back to ffmpeg for exotic formats).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright's OWN Chromium (never a system browser; required by chat_bridge).
RUN playwright install chromium --with-deps || playwright install chromium

COPY app.py chat_bridge.py tts_prompt.py ./

# Render sets $PORT at runtime; 8000 is just the local-run default.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
