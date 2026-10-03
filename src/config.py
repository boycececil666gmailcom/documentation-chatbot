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
OPENROUTER_DECISIONS_URL = os.getenv(
    "OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions"
)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash")
OPENROUTER_JEV_MODEL = os.getenv("OPENROUTER_JEV_MODEL", "typesafe/jev-1.13")
OPENROUTER_EMBED_MODEL = os.getenv(
    "OPENROUTER_EMBED_MODEL", "nvidia/nemotron-3-embed-1b:free"
)
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.2"))
OPENROUTER_PROVIDER_SORT = os.getenv("OPENROUTER_PROVIDER_SORT", "throughput")
_ignore_env = os.getenv("OPENROUTER_PROVIDER_IGNORE", "wafer")
OPENROUTER_PROVIDER_IGNORE = [p.strip() for p in _ignore_env.split(",") if p.strip()]

# PGVector Database Settings
PGVECTOR_URL = os.getenv(
    "PGVECTOR_URL",
    "postgresql+psycopg://postgres:postgrespassword123@localhost:5432/documentation_chatbot",
)
PGVECTOR_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME", "raptor_chunks")


# Chatbot Theme Settings
CHATBOT_THEME = require_env("CHATBOT_THEME")

# LangSmith Settings
LANGSMITH_TRACING = require_env("LANGSMITH_TRACING").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = require_env("LANGSMITH_PROJECT")
# endregion
