# Base = ROCm + vLLM so the FREE local model runs inside the container.
# CONFIRM the tag matches the scoring env (pod shows ROCm 7.2 + vLLM 0.16.0).
FROM rocm/vllm:latest

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# The local model is served on localhost inside the container (zero-token tier).
ENV LOCAL_BASE_URL=http://localhost:8000/v1

# CONFIRM how the scoring harness invokes the container (entrypoint vs. server,
# and whether it starts vLLM for us or expects us to). For a self-contained run,
# an entrypoint script would: (1) launch `vllm serve $LOCAL_MODEL --port 8000`,
# (2) wait for health, (3) exec the agent. Kept simple here:
ENTRYPOINT ["python", "-m", "src.agent"]
