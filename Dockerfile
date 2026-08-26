FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baileys sidecar deps (Part 7): installed at BUILD time with a pinned
# cache mount so rebuilds stay fast; runtime never touches npm.
COPY node-service/package.json node-service/package-lock.json* /app/node-service/
RUN cd /app/node-service \
    && (npm ci --omit=dev 2>/dev/null || npm install --omit=dev)

COPY . .
RUN chmod +x entrypoint.sh

ENV PYTHONPATH=/app
ENV HERMES_HOME=/root/.hermes
ENV HERMES_TIMEZONE=Asia/Kolkata

# sandy-memory plugin goes flat under plugins/ (not plugins/memory/) —
# verified against the real Hermes loader source, see plugin README.
RUN mkdir -p $HERMES_HOME/plugins \
    && cp -r plugins/memory/sandy-memory $HERMES_HOME/plugins/sandy-memory \
    && cp config.yaml $HERMES_HOME/config.yaml \
    && mkdir -p $HERMES_HOME/cron/output

# HF Docker Spaces route traffic to port 7860 by default
EXPOSE 7860

CMD ["/app/entrypoint.sh"]
