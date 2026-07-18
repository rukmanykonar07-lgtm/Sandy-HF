FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x entrypoint.sh

ENV PYTHONPATH=/app
ENV HERMES_HOME=/root/.hermes

# sandy-memory plugin goes flat under plugins/ (not plugins/memory/) —
# verified against the real Hermes loader source, see plugin README.
RUN mkdir -p $HERMES_HOME/plugins \
    && cp -r plugins/memory/sandy-memory $HERMES_HOME/plugins/sandy-memory \
    && cp config.yaml $HERMES_HOME/config.yaml \
    && mkdir -p $HERMES_HOME/cron/output

# HF Docker Spaces route traffic to port 7860 by default
EXPOSE 7860

CMD ["/app/entrypoint.sh"]
