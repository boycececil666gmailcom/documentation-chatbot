# region Module Exports
"""
RAGAS Evaluation Utilities for Documentation Chatbot.
Direct CLI tools for dataset synthesis and dynamic pipeline evaluation.
"""

from .llm_client import get_eval_models

__all__ = [
    "get_eval_models",
]
# endregion
