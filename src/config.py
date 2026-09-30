# region Config
import os

from dotenv import load_dotenv

load_dotenv()


def require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(
            f"CRITICAL CONFIG ERROR: Environment variable '{key}' is required but not set."
        )
    return val


# OpenRouter Settings
OPENROUTER_API_KEY = require_env("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731")
OPENROUTER_EMBED_MODEL = os.getenv(
    "OPENROUTER_EMBED_MODEL", "nvidia/nemotron-3-embed-1b:free"
)
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.2"))
OPENROUTER_PROVIDER_SORT = os.getenv("OPENROUTER_PROVIDER_SORT", "throughput")
_ignore_env = os.getenv("OPENROUTER_PROVIDER_IGNORE", "wafer")
OPENROUTER_PROVIDER_IGNORE = [p.strip() for p in _ignore_env.split(",") if p.strip()]

# Server Settings
BACKEND_HOST = require_env("BACKEND_HOST")
BACKEND_PORT = int(require_env("BACKEND_PORT"))

# CORS Security Settings
_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [origin.strip() for origin in _origins.split(",") if origin.strip()]
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS

# PGVector Database Settings
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "documentation_chatbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgrespassword123")

_default_pg_url = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
PGVECTOR_URL = os.getenv("PGVECTOR_URL", _default_pg_url)


# Chatbot Theme Settings
CHATBOT_THEME = require_env("CHATBOT_THEME")

# LangSmith Settings
LANGSMITH_TRACING = require_env("LANGSMITH_TRACING").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = require_env("LANGSMITH_PROJECT")
# endregion
