# Scout container (Submission 1): lean, Fireworks-only, no local model.
# Purpose: confirm the image is pullable, the /input->/output contract works,
# and see the real accuracy on the hidden tasks. linux/amd64, well under 10GB.
#
# The token-crusher container (Submission 2) will add a small local model to
# drive tokens toward zero — separate, heavier build.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src

# Scout mode: skip the local tier, route everything through the judge-provided
# Fireworks endpoint (FIREWORKS_BASE_URL / FIREWORKS_API_KEY are injected at run).
ENV USE_LOCAL=0
ENV INPUT_PATH=/input/tasks.json
ENV OUTPUT_PATH=/output/results.json

ENTRYPOINT ["python", "-m", "src.agent"]
