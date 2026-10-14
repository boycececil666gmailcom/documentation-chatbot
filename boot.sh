#!/usr/bin/env bash
set -euo pipefail

# region Utilities
SCRIPT_NAME=$(basename "$0")

log_step() {
    local step_num="$1"
    local total_steps="$2"
    local message="$3"
    echo -e "\n\033[1;96m========================================================\033[0m"
    echo -e "\033[1;92m>>> [${step_num}/${total_steps}] [${SCRIPT_NAME}] ${message}\033[0m"
    echo -e "\033[1;96m========================================================\033[0m\n"
}

# Resolve Docker CLI binary (prefers docker.exe on Windows/WSL if Linux socket is unconfigured)
resolve_docker() {
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        echo "docker"
        return
    fi
    if command -v docker.exe >/dev/null 2>&1 && docker.exe info >/dev/null 2>&1; then
        echo "docker.exe"
        return
    fi
    if command -v docker.exe >/dev/null 2>&1; then
        echo "docker.exe"
        return
    fi
    echo "docker"
}

open_browser() {
    local target_url="$1"
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command "Start-Process '${target_url}'" >/dev/null 2>&1
    elif command -v cmd.exe >/dev/null 2>&1; then
        cmd.exe /c start "" "${target_url}" >/dev/null 2>&1
    elif command -v python >/dev/null 2>&1; then
        python -m webbrowser "${target_url}" >/dev/null 2>&1
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "${target_url}" >/dev/null 2>&1
    elif command -v open >/dev/null 2>&1; then
        open "${target_url}" >/dev/null 2>&1
    else
        echo "[${SCRIPT_NAME}] Please open manually: ${target_url}"
    fi
}
# endregion

# region DockerDaemonCheck
log_step "1" "4" "Verifying and Starting Docker Daemon"

DOCKER_BIN=$(resolve_docker)

if ! "$DOCKER_BIN" info >/dev/null 2>&1; then
    echo "[${SCRIPT_NAME}] Docker daemon is not active. Starting Docker Desktop..."
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command "Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'" >/dev/null 2>&1 || true
    elif command -v cmd.exe >/dev/null 2>&1; then
        cmd.exe /c start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe" >/dev/null 2>&1 || true
    elif [ -f "/c/Program Files/Docker/Docker/Docker Desktop.exe" ]; then
        "/c/Program Files/Docker/Docker/Docker Desktop.exe" &
    fi

    echo "[${SCRIPT_NAME}] Waiting for Docker daemon to become responsive..."
    ELAPSED=0
    TIMEOUT=60
    while ! "$DOCKER_BIN" info >/dev/null 2>&1; do
        sleep 2
        ELAPSED=$((ELAPSED + 2))
        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
            echo "[${SCRIPT_NAME}] Error: Timed out waiting for Docker daemon after ${TIMEOUT}s."
            exit 1
        fi
    done
    echo "[${SCRIPT_NAME}] Docker daemon initialized successfully."
else
    echo "[${SCRIPT_NAME}] Docker daemon is active (using: ${DOCKER_BIN})."
fi
# endregion

# region ImageVerification
log_step "2" "4" "Checking Images and Pulling/Building If Missing"

PG_IMAGE="pgvector/pgvector:pg16"
CB_IMAGE="dbeaver/cloudbeaver:latest"
VLLM_IMAGE="vllm/vllm-openai:latest"
LG_IMAGE="documentation-chatbot-langgraph:latest"

# Pull remote third-party images if not present locally
if ! "$DOCKER_BIN" image inspect "${PG_IMAGE}" >/dev/null 2>&1 || \
   ! "$DOCKER_BIN" image inspect "${CB_IMAGE}" >/dev/null 2>&1 || \
   ! "$DOCKER_BIN" image inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
    echo "[${SCRIPT_NAME}] One or more base images missing from local cache. Pulling..."
    "$DOCKER_BIN" compose pull
else
    echo "[${SCRIPT_NAME}] Remote base images (${PG_IMAGE}, ${CB_IMAGE}, ${VLLM_IMAGE}) are cached locally."
fi

# Build local application container if not present
if ! "$DOCKER_BIN" image inspect "${LG_IMAGE}" >/dev/null 2>&1; then
    echo "[${SCRIPT_NAME}] LangGraph application image missing. Building container..."
    "$DOCKER_BIN" compose build
else
    echo "[${SCRIPT_NAME}] Application image (${LG_IMAGE}) is already cached locally."
fi
# endregion

# region ComposeStartup
log_step "3" "4" "Starting Docker Compose Stack and Verifying Health"

"$DOCKER_BIN" compose up -d

echo "[${SCRIPT_NAME}] Waiting for services to become responsive..."

# Wait for pgvector container health status
WAIT_SECS=0
MAX_WAIT=45
PG_READY=false
while [ "$WAIT_SECS" -lt "$MAX_WAIT" ]; do
    HEALTH_STATUS=$("$DOCKER_BIN" inspect --format='{{json .State.Health.Status}}' documentation-chatbot-pgvector 2>/dev/null || echo "\"starting\"")
    if [ "$HEALTH_STATUS" == "\"healthy\"" ]; then
        echo "[${SCRIPT_NAME}] PostgreSQL pgvector container is healthy."
        PG_READY=true
        break
    fi
    sleep 2
    WAIT_SECS=$((WAIT_SECS + 2))
done
if [ "$PG_READY" != "true" ]; then
    echo "[${SCRIPT_NAME}] Error: PostgreSQL pgvector failed to become healthy within ${MAX_WAIT}s."
    exit 1
fi

# Wait for vLLM service endpoint
WAIT_SECS=0
MAX_VLLM_WAIT=90
VLLM_READY=false
while [ "$WAIT_SECS" -lt "$MAX_VLLM_WAIT" ]; do
    if curl -s -f -o /dev/null "http://localhost:8000/health" 2>/dev/null; then
        echo "[${SCRIPT_NAME}] vLLM inference server is responsive on port 8000."
        VLLM_READY=true
        break
    fi
    sleep 3
    WAIT_SECS=$((WAIT_SECS + 3))
done
if [ "$VLLM_READY" != "true" ]; then
    echo "[${SCRIPT_NAME}] Warning: vLLM was not responsive within ${MAX_VLLM_WAIT}s. Model weights may still be loading."
fi

# Wait for LangGraph dev endpoint
WAIT_SECS=0
LG_READY=false
while [ "$WAIT_SECS" -lt "$MAX_WAIT" ]; do
    if curl -s -f -o /dev/null "http://localhost:2024/ok" 2>/dev/null; then
        echo "[${SCRIPT_NAME}] LangGraph Agent Server is responsive on port 2024."
        LG_READY=true
        break
    fi
    sleep 2
    WAIT_SECS=$((WAIT_SECS + 2))
done
if [ "$LG_READY" != "true" ]; then
    echo "[${SCRIPT_NAME}] Warning: LangGraph server on port 2024 is still initializing."
fi

# Wait for CloudBeaver web endpoint
WAIT_SECS=0
while [ "$WAIT_SECS" -lt "$MAX_WAIT" ]; do
    if curl -s -f -o /dev/null "http://localhost:8978" 2>/dev/null; then
        echo "[${SCRIPT_NAME}] CloudBeaver Web GUI is responsive on port 8978."
        break
    fi
    sleep 2
    WAIT_SECS=$((WAIT_SECS + 2))
done
# endregion

# region LaunchMonitoring
log_step "4" "4" "Launching Monitoring UIs in Browser"

# Detect dynamic tunnel URL if present in logs, else fallback to local
STUDIO_URL=$("$DOCKER_BIN" logs documentation-chatbot-langgraph 2>&1 | grep -o 'https://smith\.langchain\.com/studio/?baseUrl=https://[a-zA-Z0-9.-]*\.trycloudflare\.com' | tail -n 1 || true)
if [ -z "${STUDIO_URL}" ]; then
    STUDIO_URL="https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
fi
CLOUDBEAVER_URL="http://localhost:8978"

echo "[${SCRIPT_NAME}] Opening LangSmith Studio UI: ${STUDIO_URL}"
open_browser "${STUDIO_URL}"

sleep 1

echo "[${SCRIPT_NAME}] Opening CloudBeaver Web GUI: ${CLOUDBEAVER_URL}"
open_browser "${CLOUDBEAVER_URL}"

echo -e "\n\033[1;92m[${SCRIPT_NAME}] Boot sequence completed successfully with vLLM engine.\033[0m\n"
# endregion
