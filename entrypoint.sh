#!/bin/bash
# Hybrid process supervision:
#  - gateway (autonomous/cron, new + unattended for days) runs under
#    supervisord so an isolated crash retries silently, never touching
#    the live /chat traffic.
#  - FastAPI runs plain, exactly as it does today pre-Hermes. If it
#    crashes, the whole container exits and HF's normal restart handles
#    it — zero behavior change from current production reality.
#
# No `set -e`: FastAPI exiting non-zero (a real crash) is the normal
# path here and must NOT abort this script before the cleanup lines
# below run. Confirmed by testing — set -e caused supervisord/gateway
# to leak as orphaned processes when FastAPI crashed.

# Restore ~/.hermes/cron/jobs.json from its last Supabase backup, if any
# -- must run BEFORE supervisord/gateway below, since the gateway reads
# jobs.json on its own startup. Without this, every rebuild (including
# Sandy pushing her own self-edit) silently wiped all active mastery
# jobs. Failure here should never block boot -- log and continue.
python3 -c "import config; config.restore_hermes_jobs()" || echo "[entrypoint] jobs.json restore skipped/failed, continuing boot"

supervisord -c /app/supervisord.conf &
SUPERVISOR_PID=$!

# Gateway now logs to real files (supervisord.conf), not /dev/stdout
# directly, so Sandy's own process can read them (diagnostics.py). Tail
# both back out to this script's own stdout/stderr so HF's live log
# viewer still shows everything exactly as before -- nothing lost.
touch /tmp/hermes_gateway.log /tmp/hermes_gateway.err.log
tail -F /tmp/hermes_gateway.log &
TAIL_OUT_PID=$!
tail -F /tmp/hermes_gateway.err.log >&2 &
TAIL_ERR_PID=$!

uvicorn main:app --host 0.0.0.0 --port 7860 &
FASTAPI_PID=$!

trap 'kill -TERM $FASTAPI_PID 2>/dev/null; kill -TERM $SUPERVISOR_PID 2>/dev/null; kill -TERM $TAIL_OUT_PID $TAIL_ERR_PID 2>/dev/null' TERM INT

# Only FastAPI's exit reaches this line — a gateway crash is fully
# absorbed by supervisord above and never triggers shutdown.
wait $FASTAPI_PID
EXIT_CODE=$?
echo "[entrypoint] FastAPI exited (code $EXIT_CODE) -> shutting down gateway"
kill -TERM $SUPERVISOR_PID 2>/dev/null
kill -TERM $TAIL_OUT_PID $TAIL_ERR_PID 2>/dev/null
wait $SUPERVISOR_PID 2>/dev/null
exit $EXIT_CODE
