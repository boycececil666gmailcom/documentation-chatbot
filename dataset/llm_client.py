# region Imports & Re-exports
import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from src.llm_client import embeddings, llm

__all__ = ["llm", "embeddings"]
# endregion
