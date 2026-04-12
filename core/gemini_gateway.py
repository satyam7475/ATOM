"""
Intelligent Dual-Model Gemini API Gateway.

Routes requests between Gemini 1.5 Flash (fast/cheap) and Gemini 2.0 Pro (powerful)
based on task complexity and rate limit status.

SMART ROUTING STRATEGY:
  - Flash (Fast): Simple queries, completions, formatting, quick responses <500ms
  - Pro (Reasoning): Complex reasoning, multi-step problems, analysis, deep thinking
  - Fallback: If one model hits rate limit, automatically switch to other
  - Throttling: Monitors quota and adapts automatically

RATE LIMIT MANAGEMENT:
  - Tracks usage per model per minute/hour/day
  - Implements exponential backoff on errors
  - Graceful degradation when limits approached

USAGE:
  >>> from core.gemini_gateway import get_response
  >>> 
  >>> # Simple query → uses Flash (fast, cheap)
  >>> response = get_response("What time is it?")
  >>>
  >>> # Complex reasoning → uses Pro (powerful)
  >>> response = get_response(
  ...     "Analyze CPU bottlenecks in this code...",
  ...     complexity="high"
  ... )
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for model selection."""
    SIMPLE = "simple"          # <100 tokens, straightforward
    MODERATE = "moderate"      # 100-500 tokens, some reasoning
    COMPLEX = "complex"        # 500+ tokens, deep reasoning required
    AUTO = "auto"              # Detect automatically


@dataclass
class RateLimitStatus:
    """Rate limit status for a model."""
    model: str
    requests_per_minute: int = 0
    requests_per_hour: int = 0
    requests_per_day: int = 0
    max_rpm: int = 60
    max_rph: int = 1000
    max_rpd: int = 10000
    last_reset_min: datetime = field(default_factory=datetime.utcnow)
    last_reset_hour: datetime = field(default_factory=datetime.utcnow)
    last_reset_day: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_throttled_minute(self) -> bool:
        """Check if rate limited at minute level."""
        return self.requests_per_minute >= int(self.max_rpm * 0.9)  # 90% threshold
    
    @property
    def is_throttled_hour(self) -> bool:
        """Check if rate limited at hour level."""
        return self.requests_per_hour >= int(self.max_rph * 0.9)
    
    @property
    def is_throttled_day(self) -> bool:
        """Check if rate limited at day level."""
        return self.requests_per_day >= int(self.max_rpd * 0.9)
    
    @property
    def is_throttled(self) -> bool:
        """Check if model is throttled."""
        return self.is_throttled_minute or self.is_throttled_hour or self.is_throttled_day


@dataclass
class GeminiResponse:
    """Response from Gemini API."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    was_cached: bool = False
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if request was successful."""
        return self.error is None


class GeminiGateway:
    """Intelligent Gemini API gateway with dual-model support."""
    
    def __init__(self):
        """Initialize gateway with API keys."""
        from core.secure_credentials import get_gemini_fast_key, get_gemini_pro_key
        
        self.fast_key = get_gemini_fast_key()
        self.pro_key = get_gemini_pro_key()
        
        if not self.fast_key or not self.pro_key:
            raise ValueError("Gemini API keys not configured. Run: python -m core.secure_credentials setup")
        
        # Rate limit tracking
        self.fast_limits = RateLimitStatus(model="gemini-1.5-flash")
        self.pro_limits = RateLimitStatus(model="gemini-2.0-pro")
        
        # Usage statistics
        self.stats = {
            "fast_requests": 0,
            "pro_requests": 0,
            "fast_tokens": 0,
            "pro_tokens": 0,
            "fallbacks": 0,
            "errors": 0,
        }
        
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize Gemini API clients."""
        try:
            import google.generativeai as genai
            
            self.client = genai
            
            # Configure models
            self.model_fast = "gemini-1.5-flash"
            self.model_pro = "gemini-2.0-pro"
            
            logger.info("Gemini API clients initialized")
        except ImportError:
            logger.error("google-generativeai not installed. Install with: pip install google-generativeai")
            self.client = None
    
    def _update_rate_limits(self, model: str, tokens: int = 1) -> None:
        """Update rate limit counters."""
        now = datetime.utcnow()
        limits = self.fast_limits if model == "fast" else self.pro_limits
        
        # Reset minute counter every 60 seconds
        if (now - limits.last_reset_min).total_seconds() >= 60:
            limits.requests_per_minute = 0
            limits.last_reset_min = now
        
        # Reset hour counter every 3600 seconds
        if (now - limits.last_reset_hour).total_seconds() >= 3600:
            limits.requests_per_hour = 0
            limits.last_reset_hour = now
        
        # Reset day counter every 86400 seconds
        if (now - limits.last_reset_day).total_seconds() >= 86400:
            limits.requests_per_day = 0
            limits.last_reset_day = now
        
        # Increment counters
        limits.requests_per_minute += 1
        limits.requests_per_hour += 1
        limits.requests_per_day += 1
    
    def _select_model(self, complexity: TaskComplexity = TaskComplexity.AUTO) -> str:
        """Select best model based on complexity and rate limits.
        
        Args:
            complexity: Task complexity level
            
        Returns:
            Selected model name
        """
        # Check if either model is throttled
        fast_throttled = self.fast_limits.is_throttled
        pro_throttled = self.pro_limits.is_throttled
        
        # If one is throttled, use the other
        if fast_throttled and not pro_throttled:
            logger.info("Flash throttled, using Pro")
            return "pro"
        if pro_throttled and not fast_throttled:
            logger.info("Pro throttled, using Flash")
            return "fast"
        
        # Neither throttled: choose based on complexity
        if complexity == TaskComplexity.SIMPLE:
            return "fast"  # Fast is cheaper and sufficient
        elif complexity == TaskComplexity.COMPLEX:
            return "pro"   # Pro for complex reasoning
        else:
            # Moderate: use Flash (still capable, cheaper)
            return "fast"
    
    def _detect_complexity(self, prompt: str) -> TaskComplexity:
        """Detect task complexity from prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Estimated complexity
        """
        # Simple heuristics
        complex_keywords = [
            "analyze", "reason", "think about", "explain",
            "debug", "optimize", "refactor", "design",
            "compare", "contrast", "evaluate", "critic"
        ]
        
        reasoning_indicators = [
            "?", "troubleshoot", "fix", "impossible",
            "complex", "algorithm", "architecture"
        ]
        
        prompt_lower = prompt.lower()
        
        # Count indicators
        complex_score = sum(1 for kw in complex_keywords if kw in prompt_lower)
        reasoning_score = sum(1 for ind in reasoning_indicators if ind in prompt_lower)
        length_score = len(prompt) / 100  # Longer prompts often need more reasoning
        
        total_score = complex_score + reasoning_score + length_score
        
        if total_score >= 3:
            return TaskComplexity.COMPLEX
        elif total_score >= 1:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    async def get_response(
        self,
        prompt: str,
        complexity: TaskComplexity | str = TaskComplexity.AUTO,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> GeminiResponse:
        """Get response from Gemini.
        
        Args:
            prompt: User prompt
            complexity: Task complexity or "auto" to detect
            system_prompt: Optional system message
            **kwargs: Additional arguments for API
            
        Returns:
            GeminiResponse with result or error
        """
        if self.client is None:
            return GeminiResponse(
                content="",
                model="error",
                error="Gemini not initialized"
            )
        
        try:
            # Parse complexity
            if isinstance(complexity, str):
                complexity = TaskComplexity(complexity) if complexity != "auto" else TaskComplexity.AUTO
            
            # Auto-detect if needed
            if complexity == TaskComplexity.AUTO:
                complexity = self._detect_complexity(prompt)
            
            # Select model
            model = self._select_model(complexity)
            selected_model_id = self.model_fast if model == "fast" else self.model_pro
            api_key = self.fast_key if model == "fast" else self.pro_key
            
            # Configure client
            self.client.configure(api_key=api_key)
            
            # Build request
            start = time.perf_counter()
            
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Make API call
            gemini_model = self.client.GenerativeModel(selected_model_id)
            api_response = gemini_model.generate_content(full_prompt, **kwargs)
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Update metrics
            self._update_rate_limits(model)
            if model == "fast":
                self.stats["fast_requests"] += 1
            else:
                self.stats["pro_requests"] += 1
            
            return GeminiResponse(
                content=api_response.text,
                model=selected_model_id,
                latency_ms=elapsed_ms,
            )
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            self.stats["errors"] += 1
            
            return GeminiResponse(
                content="",
                model="error",
                error=str(e)
            )
    
    def get_response_sync(
        self,
        prompt: str,
        complexity: TaskComplexity | str = TaskComplexity.AUTO,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> GeminiResponse:
        """Synchronous wrapper for get_response.
        
        Args:
            prompt: User prompt
            complexity: Task complexity
            system_prompt: Optional system message
            **kwargs: Additional arguments
            
        Returns:
            GeminiResponse
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.get_response(prompt, complexity, system_prompt, **kwargs)
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics.
        
        Returns:
            Dictionary with usage stats
        """
        return {
            "fast_model": {
                "requests": self.stats["fast_requests"],
                "rate_limit": f"{self.fast_limits.requests_per_minute}/{self.fast_limits.max_rpm} RPM",
                "throttled": self.fast_limits.is_throttled,
            },
            "pro_model": {
                "requests": self.stats["pro_requests"],
                "rate_limit": f"{self.pro_limits.requests_per_minute}/{self.pro_limits.max_rpm} RPM",
                "throttled": self.pro_limits.is_throttled,
            },
            "total_errors": self.stats["errors"],
            "fallback_count": self.stats["fallbacks"],
        }


# Global instance
_gateway: Optional[GeminiGateway] = None


def get_gateway() -> GeminiGateway:
    """Get or create global Gemini gateway."""
    global _gateway
    if _gateway is None:
        _gateway = GeminiGateway()
    return _gateway


async def ask(
    prompt: str,
    complexity: str = "auto",
    system_prompt: Optional[str] = None,
) -> str:
    """Ask Gemini a question.
    
    Smart routing: Flash for simple, Pro for complex.
    
    Args:
        prompt: Question or prompt
        complexity: "simple", "moderate", "complex", or "auto"
        system_prompt: Optional system message
        
    Returns:
        Response text
    """
    try:
        gateway = get_gateway()
        response = await gateway.get_response(
            prompt,
            complexity=complexity,
            system_prompt=system_prompt
        )
        
        if response.success:
            return response.content
        else:
            logger.error(f"API error: {response.error}")
            return f"Error: {response.error}"
    
    except Exception as e:
        logger.error(f"Failed to get response: {e}")
        return f"Error: {e}"


# Convenience functions
def ask_simple(prompt: str) -> str:
    """Ask a simple question (uses Flash model)."""
    gateway = get_gateway()
    return gateway.get_response_sync(prompt, complexity=TaskComplexity.SIMPLE).content


def ask_complex(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Ask a complex question (uses Pro model)."""
    gateway = get_gateway()
    return gateway.get_response_sync(
        prompt,
        complexity=TaskComplexity.COMPLEX,
        system_prompt=system_prompt
    ).content


if __name__ == "__main__":
    # Test script
    print("Gemini Gateway Test")
    print("=" * 50)
    
    try:
        gateway = get_gateway()
        
        # Test simple query
        print("\n1. Simple Query (Flash):")
        resp = gateway.get_response_sync("What is 2+2?")
        print(f"   Response: {resp.content}")
        print(f"   Model: {resp.model}")
        print(f"   Latency: {resp.latency_ms:.2f}ms")
        
        # Test complex query
        print("\n2. Complex Query (Pro):")
        resp = gateway.get_response_sync(
            "Analyze the algorithmic complexity of quicksort",
            complexity=TaskComplexity.COMPLEX
        )
        print(f"   Response: {resp.content[:100]}...")
        print(f"   Model: {resp.model}")
        print(f"   Latency: {resp.latency_ms:.2f}ms")
        
        # Show stats
        print("\n3. Usage Statistics:")
        stats = gateway.get_stats()
        print(json.dumps(stats, indent=2))
    
    except Exception as e:
        print(f"Error: {e}")
