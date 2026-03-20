"""
ATOM v14 -- Router package.

Re-exports Router and compress_query for backward compatibility.
"""

from .router import Router, compress_query

__all__ = ["Router", "compress_query"]
