# ==============================================================================
# Core RAG Backend Engine Service Dockerfile
# ==============================================================================
FROM python:3.12-slim

#region UV CLI Setup
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
#endregion

WORKDIR /app

#region User Security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser
#endregion

#region Install Dependencies
ENV UV_PROJECT_ENVIRONMENT=/usr/local
COPY --chown=appuser:appgroup pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project
#endregion

#region Application Setup
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup langgraph.json /app/
COPY --chown=appuser:appgroup infra/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENV PYTHONUNBUFFERED=1
EXPOSE 8000 2024
#endregion

USER root

CMD ["/app/entrypoint.sh"]
