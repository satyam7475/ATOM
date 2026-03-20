"""ATOM prompt + local LLM bridge (offline-first)."""

from .local_brain_controller import LocalBrainController
from .structured_prompt_builder import StructuredPromptBuilder

__all__ = ["LocalBrainController", "StructuredPromptBuilder"]
