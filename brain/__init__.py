"""
Brain package — production modules only.

Active modules:
  mlx_llm      — MLX local LLM inference (dual-role: fast 1.7B + primary 4B)
  memory_graph  — Long-term memory graph with persistence
  mini_llm      — Auxiliary GGUF inference (benchmarks / tooling)
"""
from .memory_graph import MemoryGraph, MemoryNode

__all__ = [
    'MemoryGraph', 'MemoryNode',
]
