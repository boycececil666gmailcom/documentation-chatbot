# region Imports
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# endregion


# region Types
class UserMemoryProfile:
    """Represents persistent long-term memory for a user or session."""

    def __init__(self, user_id: str, preferences: dict[str, Any], learned_facts: list[str]) -> None:
        self.user_id = user_id
        self.preferences = preferences
        self.learned_facts = learned_facts

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "learned_facts": self.learned_facts,
        }


# endregion


# region Scaffolding
_MEMORY_FILE = Path("dataset/long_term_memory.json")


def _get_memory_path() -> Path:
    """Ensures parent directory exists and returns memory file path."""
    if not _MEMORY_FILE.parent.exists():
        _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    return _MEMORY_FILE


def _load_all_memories() -> dict[str, dict[str, Any]]:
    """Loads all persisted memory profiles from disk."""
    path = _get_memory_path()
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_all_memories(data: dict[str, dict[str, Any]]) -> None:
    """Persists memory profiles to disk."""
    path = _get_memory_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[Memory-Save] Warning: Could not save memory: {exc}")


# endregion


# region Memory
class LongTermMemoryManager:
    """Manages persistent long-term knowledge, user preferences, and learned technical facts."""

    @classmethod
    def load_user_memory(cls, user_id: str = "default_user") -> dict[str, Any]:
        """Loads user memory profile including preferences and learned facts."""
        memories = _load_all_memories()
        if user_id not in memories:
            # Seed default profile with domain preferences
            memories[user_id] = {
                "user_id": user_id,
                "preferences": {
                    "preferred_language": "C++ / QML",
                    "code_samples_requested": True,
                    "depth": "detailed_technical",
                },
                "learned_facts": [
                    "User develops embedded automotive UI applications with Kanzi.",
                    "Performance optimization, memory budgets, and draw calls are critical constraints.",
                ],
            }
            _save_all_memories(memories)
        return memories[user_id]

    @classmethod
    def record_fact(cls, user_id: str, fact: str) -> None:
        """Appends a new verified technical fact to the user's persistent profile."""
        memories = _load_all_memories()
        profile = memories.setdefault(user_id, {
            "user_id": user_id,
            "preferences": {},
            "learned_facts": [],
        })
        if fact not in profile["learned_facts"]:
            profile["learned_facts"].append(fact)
            _save_all_memories(memories)


class ShortTermMemoryManager:
    """Formats and summarizes active conversation message history."""

    @staticmethod
    def extract_recent_history(messages: list[BaseMessage], max_turns: int = 4) -> str:
        """Formats the last N conversation turns into structured context."""
        if not messages:
            return "No previous conversation turns."
        recent = messages[-max_turns:]
        formatted_turns: list[str] = []
        for msg in recent:
            role = "User" if isinstance(msg, HumanMessage) else ("Assistant" if isinstance(msg, AIMessage) else "System")
            content_preview = str(msg.content)[:200].replace("\n", " ")
            formatted_turns.append(f"{role}: {content_preview}")
        return "\n".join(formatted_turns)


# endregion


# region Core
def get_memory_context(user_id: str, messages: list[BaseMessage]) -> tuple[dict[str, Any], str]:
    """Primary declarative memory fetch returning long-term profile and short-term conversation context."""
    lt_memory = LongTermMemoryManager.load_user_memory(user_id)
    st_context = ShortTermMemoryManager.extract_recent_history(messages)
    return lt_memory, st_context


# endregion

