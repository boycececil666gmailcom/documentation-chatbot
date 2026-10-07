# region Build & Setup
FROM python:3.12-slim

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /bin/uvx


WORKDIR /app

# Enable byte-code compilation and install dependencies into system environment
ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=WARNING

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project --no-dev

# Copy application files
COPY src/ /app/src/
COPY langgraph.json /app/langgraph.json
# endregion

# region Runtime Configuration
EXPOSE 2024

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024", "--no-browser", "--server-log-level", "WARNING"]
# endregion
