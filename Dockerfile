FROM python:3.11-slim

# ---- Critical for HF Spaces ----
# Only /tmp is writable. Every library that caches models/data
# MUST be pointed at /tmp explicitly, or it will crash-loop or
# get stuck "Restarting" on the platform.
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    XDG_CACHE_HOME=/tmp/cache \
    HOME=/tmp \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY app/ ./app/

# HF Spaces expects the app to listen on port 7860 by default
EXPOSE 7860

# Non-root user (HF Spaces runs containers as non-root anyway,
# but being explicit avoids permission surprises)
RUN useradd -m -u 1000 sandy && chown -R sandy /app /tmp
USER sandy

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
