"""
ATOM -- Agentic Brain Controller (LLM + Tool Use + ReAct Loop).

The brain of ATOM. Runs a local LLM that can:
  1. Respond with natural language (conversation)
  2. Call tools to perform system actions (tool use)
  3. See tool results and decide next actions (ReAct loop)
  4. Chain multiple actions for complex requests

Flow:
  Query -> Build Prompt -> LLM generates response
    -> Parse for tool calls
      -> If tool calls found: execute via ActionExecutor, collect observations
         -> Feed observations back to LLM (up to MAX_REACT_STEPS)
      -> If no tool calls: emit text response for TTS

This is what makes ATOM a JARVIS-level system instead of a regex remote control.
The LLM REASONS about what to do, not just pattern-match.

Event contract:
  Emits: partial_response, cursor_response, metrics_latency, metrics_event,
         tool_executed, plan_started, plan_step_complete
  On error: response_ready, llm_error
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import time
import uuid
from collections import deque
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from core.query_policy import (
    detect_response_language,
    ResponseMode,
    classify_response_mode,
    normalize_query,
    should_export_report,
    slugify_query,
    summarize_report,
)
from core.reasoning.tool_parser import parse_tool_calls
from core.runtime.v7_context import V7RuntimeContext

if TYPE_CHECKING:
    from core.async_event_bus import AsyncEventBus
    from core.brain_mode_manager import BrainModeManager
    from core.reasoning.action_executor import ActionExecutor
    from cursor_bridge.structured_prompt_builder import StructuredPromptBuilder

logger = logging.getLogger("atom.local_brain")

_SENTENCE_BOUNDARY = re.compile(r"[.!?]\s")
_SENTENCE_END = re.compile(r"[.!?]$")
_INLINE_TRACE_RE = re.compile(r"^(?:[a-z_]+\([^)]{0,200}\)\s*[→:=-]+\s*)+", re.I)
_TRANSCRIPT_SPLIT_RE = re.compile(r"\b(?:User|Boss|ATOM|Assistant):", re.I)
_TRANSCRIPT_LABEL_RE = re.compile(r"\b(?:User|Boss|ATOM|Assistant):\s*", re.I)
_REPEATED_SPEAKER_RE = re.compile(r"^(?:(?:User|Boss|ATOM|Assistant)\s*:?\s*)+$", re.I)

# Hard-stop ChatML / HF tokens that must never reach TTS. We split the text
# at the first occurrence (truncating everything after) and also separately
# strip any stray copies anywhere in the body.
_STRIP_HARD_STOP_TOKENS_RE = re.compile(
    r"<\|(?:endoftext|im_end|im_start|user|assistant|system|eot_id|"
    r"begin_of_text|end_of_text|start_header_id|end_header_id|"
    r"reserved_special_token_\d+)\|>",
    re.I,
)
# Mirror of brain/mlx_llm.py:_LEADING_QUOTE_WRAP_RE — catches lone leading
# quotes followed by Boss/Sir/lowercase/apostrophe so a streamed clause like
# `" haven't set an alarm.` or `"'m sorry, Boss.` never reaches TTS.
_LEADING_QUOTE_WRAP_RE = re.compile(
    r"""^\s*[\"\u201c\u201d\u2018\u2019`]+\s*"""
    r"""(?="""
    r"""(?:Boss|Satyam|Sir|Ma'am|Madam|Hey|OK|Okay|Alright)\b"""
    r"""|[A-Z]"""
    r"""|'[a-z]"""
    r"""|[a-z]"""
    r""")""",
    re.U,
)
_TRAILING_UNCLOSED_QUOTE_RE = re.compile(r"""[\"\u201c\u201d\u2018\u2019]+\s*$""")
_STRIP_SPECIAL_TOKENS_RE = re.compile(
    r"<\|(?:endoftext|im_end|im_start|user|assistant|system|eot_id|"
    r"begin_of_text|end_of_text|start_header_id|end_header_id|"
    r"reserved_special_token_\d+)\|>",
    re.I,
)
# Anything after a fresh role header (Human:, User:, Boss:) is the next-turn
# prompt that the model is hallucinating; truncate at the FIRST occurrence
# so we keep the assistant's own reply but drop the leaked next-turn turn.
# We require either a leading newline or a beginning-of-string anchor so we
# don't accidentally chop mid-sentence references like "the user: said ...".
_STRIP_TRANSCRIPT_HEADERS_RE = re.compile(
    r"(?:^|[\r\n])\s*(?:Human|User|Boss):\s",
    re.I,
)
# Broadened: also flag chain-of-thought prefaces and third-person narration so
# the quality-reject check can drop them. The previous regex only anchored on
# a handful of instruction echoes ("direct answer", "strict output recovery");
# real model leaks include "Okay, let's see …" and "So the user is asking …"
# which we now catch explicitly.
# Quoted-prefix tolerance: the leak we keep seeing in the wild looks like
#   "Dear Boss" — the user is greeting you, so respond politely and warmly.
# i.e. the model first quotes the user's transcribed text and *then* narrates
# what it should do. The quoted prefix would otherwise let the leak slip past
# any anchored "starts-with" check, so this optional group consumes a leading
# quoted phrase plus a separator (em/en dash, double-dash, colon, comma, etc.)
# before the actual instruction-echo token.
_QUOTED_USER_ECHO_PREFIX = (
    r"(?:[\"'\u201c\u2018`]"        # opening quote (straight, smart, backtick)
    r"[^\"'\u201c\u201d\u2018\u2019`\n]{1,80}"   # quoted body (single line)
    r"[\"'\u201d\u2019`]"           # closing quote
    r"\s*[\u2013\u2014:,\-]+\s*)?"  # optional dash/colon/comma separator + space
)
_INSTRUCTION_ECHO_RE = re.compile(
    r"^"
    + _QUOTED_USER_ECHO_PREFIX +
    r"(?:"
    r"the final answer should|reply with|direct answer|current user request|"
    r"strict output recovery|response contract|the user is asking|this is a|"
    r"boss explicitly asked|"
    # Direct narration variants seen in production logs.
    r"the user (?:is|has|was)\s+(?:greeting|asking|requesting|wanting|saying|trying|having|expressing)\b|"
    r"the user wants\b|the user said\b|the user just\b|"
    r"so respond\b|respond politely\b|respond warmly\b|respond gently\b|respond empathetically\b|"
    # CoT / stall prefaces below.
    r"okay,?\s+(?:let'?s|lets|let me|the\s+user)\b|"
    r"alright,?\s+(?:so|let'?s|lets|let me|the\s+user)\b|"
    r"well,?\s+(?:so|let'?s|lets|let me|the\s+user)\b|"
    r"let'?s\s+(?:see|think|break|try|start|check|verify|look|reason|figure)\b|"
    r"let\s+me\s+(?:think|see|try|consider|check|verify|look|reason|figure|process|recall|prepare)\b|"
    r"hmm+,?|um+,?|uh+,?|"
    r"(?:so,?\s+)?the\s+(?:question|query|request|issue|problem)\s+is\b|"
    r"the\s+user\s+(?:is|has|was)?\s*(?:asking|wants|says|said|greeting|requesting|wanting|trying|having|expressing)\b|"
    r"i\s+(?:should|need\s+to|have\s+to|must|will|am\s+going\s+to)\s+"
    r"(?:think|consider|figure|reason|recall|acknowledge|show|offer|respond|focus|make\s+sure|remember|note|check|verify|look|process|prepare)\b|"
    # \"My role is to respond as ATOM.\" / \"My job is to ...\"
    r"my\s+(?:role|job|task|goal)\s+(?:is|here\s+is)\s+(?:to\s+)?\b|"
    # \"In the current context,\" / \"From my memory,\" / \"From the conversation history,\"
    r"(?:in|from|within)\s+(?:the\s+)?(?:current\s+)?(?:context|conversation(?:\s+history)?|memory|chat\s+history|transcript|history)\b|"
    # \"Keep it concise / friendly / professional.\"
    r"keep\s+(?:it|the\s+(?:answer|response|reply|tone))\s+(?:concise|brief|short|simple|friendly|warm|professional|polite|casual|natural)\b"
    r")",
    re.I,
)
_IMPERATIVE_ECHO_RE = re.compile(
    r"^(?:explain|compare|describe|tell me|answer|summarize|give)\b",
    re.I,
)
_MEMORY_ACK_RE = re.compile(r'^"?((?:yes,\s*)?i remember (?:it|that))\.?"?$', re.I)

# Strip leading chain-of-thought prefaces from any emittable sentence before
# it reaches TTS. Mirrors the MLX-side stripper but runs on the controller
# output path so non-MLX fallbacks (e.g. cloud responses) also stay clean.
_COT_PREFACE_STRIP_RE = re.compile(
    r"""
    ^\s*
    # Optional leading quoted user-text + dash/colon separator. Keeps the
    # stripper aligned with the rejector so a "Dear Boss" — narration leak
    # is peeled cleanly even when the rejector decides to keep it.
    (?:
        ["'\u201c\u2018`]
        [^"'\u201c\u201d\u2018\u2019`\n]{1,80}
        ["'\u201d\u2019`]
        \s*[\u2013\u2014:,\-]+\s*
    )?
    (?:
        (?:okay|ok|alright|well|so|hmm+|um+|uh+)\b[,.!]?\s*
        (?:let(?:'|\u2019)?s?\s+(?:see|think|break|try|start|go|check|verify|look|reason|figure)\b[^.?!]*[.?!]\s*)?
      |
        let(?:'|\u2019)?s?\s+(?:see|think|break|try|start|go|check|verify|look|reason|figure)\b[^.?!]*[.?!]\s*
      |
        # \"Let me check my memory.\" / \"Let me reason about this.\"
        let\s+me\s+(?:think|see|try|consider|check|verify|look|reason|figure|process|recall|prepare)\b[^.?!]*[.?!]\s*
      |
        (?:the\s+user|boss|the\s+speaker)\s+(?:is\s+|has\s+|was\s+)?
        (?:greeting|asking|wants|says|said|needs|wondering|requesting|trying|having|expressing)
        [^.?!]*[.?!]\s*
      |
        (?:so\s+)?(?:the\s+)?(?:question|query|request|issue|problem)\s+is\b[^.?!]*[.?!]\s*
      |
        i\s+(?:should|need\s+to|have\s+to|must|will|am\s+going\s+to)\s+
        (?:think|consider|figure|reason|recall|acknowledge|show|offer|respond|respond\s+with|focus|make\s+sure|remember|note|check|verify|look|process|prepare)
        \b[^.?!]*[.?!]\s*
      |
        # \"My role is to respond as ATOM.\" / \"My job is to ...\"
        my\s+(?:role|job|task|goal)\s+(?:is|here\s+is)\s+(?:to\s+)?[^.?!]*[.?!]\s*
      |
        # \"In the current context, there's no mention of...\"
        (?:in|from|within)\s+
        (?:the\s+)?
        (?:current\s+)?
        (?:context|conversation(?:\s+history)?|memory|chat\s+history|transcript|history)
        [,]?\s+[^.?!]*[.?!]\s*
      |
        # \"Keep it concise and friendly.\" — instruction echo to self.
        keep\s+(?:it|the\s+(?:answer|response|reply|tone))\s+(?:concise|brief|short|simple|friendly|warm|professional|polite|casual|natural)
        [^.?!]*[.?!]\s*
      |
        (?:hmm+|um+|uh+|er+|ah+)[,.!]?\s+
      |
        so\s+respond\s+(?:politely|warmly|briefly|kindly|directly|gently|empathetically)[^.?!]*[.?!]\s*
    )+
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


# Streaming-safe variant: same patterns but accepts comma OR end-of-string
# as the terminator. The base stripper above demands sentence-ending
# punctuation [.?!] which never appears mid-stream when the controller
# flushes at the FIRST clause boundary (a comma). Without this, a leak
# like
#       "Yeah Boss" — the user is greeting you,    <-- emitted at clause
#       respond politely and warmly.               <-- next slice
# slips past the sanitiser one fragment at a time and ATOM speaks its own
# instructions out loud. We run this *only* on text that does NOT end in
# [.?!] so well-formed responses keep their normal handling.
_COT_PREFACE_STRIP_PARTIAL_RE = re.compile(
    r"""
    ^\s*
    (?:
        ["'\u201c\u2018`]
        [^"'\u201c\u201d\u2018\u2019`\n]{1,80}
        ["'\u201d\u2019`]
        \s*[\u2013\u2014:,\-]+\s*
    )?
    (?:
        (?:the\s+user|boss|the\s+speaker)\s+(?:is\s+|has\s+|was\s+)?
        (?:greeting|asking|wants|says|said|needs|wondering|requesting|trying|having|expressing)
        [^.?!]*(?:[,]\s*|$)
      |
        so\s+respond\s+(?:politely|warmly|briefly|kindly|directly|gently|empathetically)
        [^.?!]*(?:[,]\s*|$)
      |
        respond\s+(?:politely|warmly|briefly|kindly|directly|gently|empathetically)
        [^.?!]*(?:[,]\s*|$)
      |
        # \"Okay, the user is asking ...,\" mid-flush leak.
        (?:okay|ok|alright|well|so)\b[,.!]?\s+
        (?:the\s+user|boss|the\s+speaker)\s+(?:is\s+|has\s+|was\s+)?
        (?:greeting|asking|wants|says|said|needs|wondering|requesting|trying|having|expressing)
        [^.?!]*(?:[,]\s*|$)
      |
        # \"Let me check my memory,\" / \"Let me think about that,\"
        let\s+me\s+(?:think|see|try|consider|check|verify|look|reason|figure|process|recall|prepare)
        [^.?!]*(?:[,]\s*|$)
      |
        # \"My role is to respond as ATOM,\" mid-flush.
        my\s+(?:role|job|task|goal)\s+(?:is|here\s+is)\s+(?:to\s+)?
        [^.?!]*(?:[,]\s*|$)
      |
        # \"I need to acknowledge their difficulty,\" mid-flush.
        i\s+(?:should|need\s+to|have\s+to|must|will|am\s+going\s+to)\s+
        (?:think|consider|figure|reason|recall|acknowledge|show|offer|respond|focus|make\s+sure|remember|note|check|verify|look|process|prepare)
        [^.?!]*(?:[,]\s*|$)
    )+
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


# Stage-direction parenthetical sanitiser. Definition lives in the
# shared brain._speech_sanitizer module so the streaming TTS path,
# the batch LLM path and the LocalBrainController hot-text path all
# use the same regex. Sprint A3 unified the three duplicate copies.
from brain._speech_sanitizer import (  # noqa: E402
    strip_stage_direction_leak as _strip_stage_direction_leak,
)


def _strip_cot_preface(text: str) -> str:
    """Peel chain-of-thought / stall prefaces from the head of an emittable
    sentence. Safe on empty input and idempotent across repeated calls.

    Runs the strict (sentence-terminated) stripper first, then a streaming
    fallback that tolerates comma terminators / end-of-string. Together
    these catch both the well-formed final-text leak and the per-clause
    streamed leak that small instruction-tuned models emit.
    """
    if not text:
        return text
    out = _strip_stage_direction_leak(text)
    prev = None
    for _ in range(3):
        if out == prev:
            break
        prev = out
        out = _COT_PREFACE_STRIP_RE.sub("", out, count=1).lstrip()
        # Stream fallback: only run when nothing was stripped above (so we
        # don't double-process well-formed text) AND the head still looks
        # like a narration-leak fragment ending mid-clause.
        if out == prev:
            cand = _COT_PREFACE_STRIP_PARTIAL_RE.sub("", out, count=1).lstrip()
            if cand != out:
                out = cand
        # Re-run stage-direction strip in case CoT removal exposed a
        # nested leak like "Okay. (calmly) Boss, …".
        out = _strip_stage_direction_leak(out)
    return out


def _looks_like_pure_instruction_leak(text: str) -> bool:
    """Detect responses that are ENTIRELY narration about the user, with
    no actual answer content. Used as a final ``hard reject`` so we never
    emit them — even partially — to TTS.

    Also catches *unfinished* narration leaks (no respond-clause yet) when
    the upstream sanitiser has already started flushing token-streamed
    clauses and the leak is being shipped one comma at a time.
    """
    if not text:
        return False
    head = text.strip()
    if not head:
        return False
    # Trim a single leading quoted-prefix if present so we can match the
    # narration directly: '"Yeah Boss" — the user is greeting you' style.
    head = re.sub(
        r'^\s*["\u201c\u2018\'`][^"\u201c\u201d\u2018\u2019\'`\n]{1,80}'
        r'["\u201d\u2019\'`]\s*[\u2013\u2014:,\-]+\s*',
        "",
        head,
    )
    # Also tolerate a half-stripped quote (e.g. ``Yeah Boss" — ...``)
    # produced when an earlier pass called .strip('"').
    head = re.sub(
        r'^\s*[A-Za-z][^"\u201c\u201d\u2018\u2019\'`\n]{0,79}'
        r'["\u201d\u2019\'`]\s*[\u2013\u2014:,\-]+\s*',
        "",
        head,
    )
    head = head.lstrip(' "\u201c\u2018\'`-:>')
    full_form = re.match(
        r"^(?:the\s+user|boss|the\s+speaker)\s+(?:is\s+|has\s+|was\s+)?"
        r"(?:greeting|asking|wants|says|said|needs|wondering|requesting|trying|having|expressing)"
        r"[^.?!]*?,?\s*"
        r"(?:so\s+)?respond\s+"
        r"(?:politely|warmly|briefly|kindly|directly|gently|empathetically)\b",
        head,
        re.IGNORECASE,
    )
    if full_form is not None:
        return True
    # Unfinished narration ("the user is greeting you" with no
    # respond-clause yet) is still definitively a leak when it appears at
    # the start of an emittable clause. Real answers don't open with
    # third-person narration about Boss.
    partial_narration = re.match(
        r"^(?:the\s+user|boss|the\s+speaker)\s+(?:is\s+|has\s+|was\s+)?"
        r"(?:greeting|asking|wants|says|said|needs|wondering|requesting|trying|having|expressing)\b",
        head,
        re.IGNORECASE,
    )
    if partial_narration is not None:
        return True
    # \"My role is to respond as ATOM.\" — only flag when the ENTIRE
    # head is that single clause. If the model self-narrates and THEN
    # gives a real answer, we want the stripper to peel the preface and
    # keep the answer; we should NOT drop the whole emission here.
    role_narration_only = re.fullmatch(
        r"\s*my\s+(?:role|job|task|goal)\s+(?:is|here\s+is)\s+(?:to\s+)?"
        r"[^.?!]*[.?!]?\s*",
        head,
        re.IGNORECASE,
    )
    if role_narration_only is not None:
        return True
    # \"Let me check my memory.\" / \"Let me think about that.\" — only flag
    # when the ENTIRE response is that CoT clause, not when a real answer
    # follows. We test this by stripping the CoT preface and checking if
    # what remains is empty / sub-3-words.
    let_me_only = re.fullmatch(
        r"\s*let\s+me\s+(?:think|see|try|consider|check|verify|look|reason|figure|process|recall|prepare)\b"
        r"[^.?!]*[.?!]?\s*",
        head,
        re.IGNORECASE,
    )
    if let_me_only is not None:
        return True
    # \"Okay, the user is asking ...\" with no actual reply attached — only
    # flag when the entire head is the CoT opener clause.
    cot_opener_only = re.fullmatch(
        r"\s*(?:okay|ok|alright|well|so)\b[,.!]?\s+"
        r"(?:the\s+user|let\s+me|let'?s|i\s+(?:should|need\s+to|will|am\s+going\s+to))"
        r"[^.?!]*[.?!]?\s*",
        head,
        re.IGNORECASE,
    )
    return cot_opener_only is not None


# Declarative reasoning-leak sentences produced by small instruction-tuned
# models. These don't match the CoT preface regex (they're not "preface +
# answer"; they're the whole reply being the thought). Observed in logs:
#
#   - "Based on the response contract, I should confirm my activity ..."
#   - "The user previously asked if I was active properly."
#   - "Now they're asking about what happened to me."
#   - "First, looking at the conversation history ..."
#   - "I need to help the user find good wallpaper examples ..."
#   - "I don't need to use any tools here since it's straightforward."
#   - "Just respond with the standard message."
#   - "Wait, looking at the available tools, there's spotlight_search ..."
#   - "Since I can't browse the internet directly, I should rely on ..."
#   - "The answer needs to be concise and in plain text."
#   - '" or similar.'  (dangling quote-tail)
#
# We match the whole sentence (not just a prefix). Real JARVIS-style
# answers don't talk about "the response contract", "the user previously
# asked", "looking at the conversation history", or "I need to help the
# user".
_REASONING_LEAK_SENTENCE_RE = re.compile(
    r"""
    ^\s*
    (?:
        # Self-referential plan / obligation
        (?:based\s+on\s+(?:the\s+)?(?:response\s+contract|conversation\s+history|available\s+tools|context|memory|system\s+prompt))\b |
        (?:according\s+to\s+(?:the\s+)?(?:response\s+contract|conversation\s+history|system\s+prompt|context))\b |
        # Third-person narration ABOUT the user
        (?:the\s+user\s+(?:previously\s+)?(?:asked|wants|wanted|requested|said|is\s+asking|is\s+requesting|was\s+asking|has\s+asked))\b |
        (?:now\s+(?:they|the\s+user)['\u2019]?(?:re|s)?\s+asking)\b |
        (?:they['\u2019]re\s+asking\s+(?:about|for|if))\b |
        # Meta-instruction / self-direction
        (?:i\s+(?:should|need\s+to|have\s+to|must|will|am\s+going\s+to|am\s+supposed\s+to)\s+
            (?:confirm|help\s+the\s+user|respond|reply|answer|think|consider|check|look\s+at|figure\s+out|
               provide|give|offer|rely|use|find|suggest|explain|start|begin|acknowledge|keep\s+it|make\s+sure)
        )\b |
        (?:i\s+don['\u2019]?t\s+need\s+to\s+use\s+any\s+tools)\b |
        (?:since\s+i\s+can['\u2019]?t\s+(?:browse|access|use))\b |
        # "Just respond with X." / "Just stick to ..."
        (?:just\s+(?:respond|stick|use|go\s+with|keep\s+it|keep\s+the))\b |
        # "First," / "Second," / "Wait," — stream narration markers
        (?:first\s*,\s+(?:i|the\s+user|looking\s+at|let|we))\b |
        (?:wait\s*,\s+(?:looking\s+at|i|the\s+user))\b |
        # Meta-sentence "The answer needs to be ..."
        (?:the\s+(?:answer|response|reply)\s+(?:needs\s+to\s+be|should\s+be|must\s+be)\s+
           (?:concise|brief|short|simple|friendly|warm|professional|polite|casual|natural|in\s+plain\s+text))\b |
        # Explicit "response contract" mention — always a leak
        (?:[^.?!]*response\s+contract)\b |
        # Looking at / based on / referring to ... history/tools/context
        (?:(?:looking\s+at|referring\s+to)\s+(?:the\s+)?(?:conversation\s+history|available\s+tools|context|memory|system\s+prompt))\b
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# Lone-quote tail fragment: "\" or similar.", '"' alone, '" right.' etc.
# A real answer never starts with a bare closing quote and 1-3 words.
_LONE_QUOTE_TAIL_RE = re.compile(
    r"""^\s*["\u201c\u201d\u2018\u2019`]+\s*[A-Za-z][^"\u201c\u201d\u2018\u2019`\n]{0,30}[.?!]?\s*$""",
    re.IGNORECASE,
)


def _looks_like_reasoning_leak(text: str) -> bool:
    """Return True when a whole sentence is the model reasoning aloud.

    Drops the sentence before it reaches TTS. Callers ALREADY ran
    `_strip_cot_preface` first, so only declarative leaks (the ones that
    are not preface + content) land here.
    """
    if not text:
        return False
    head = text.strip()
    if not head:
        return False
    # Short sentences (< 3 words) that are just a quoted tail are a leak.
    if _LONE_QUOTE_TAIL_RE.match(head):
        word_count = len(head.split())
        if word_count <= 4:
            return True
    if _REASONING_LEAK_SENTENCE_RE.match(head):
        return True
    return False


# v3 prompt-text-leak fingerprint detector.
#
# The Qwen3-8B model (and any small instruction-tuned model under tight
# max_tokens) sometimes regurgitates literal lines from its own system
# prompt as the answer. The atomlogs.txt session showed it speaking
# verbatim: "the final answer only.", "One short line.",
# "if the question is a simple, short, or info query, give one short
#  sentence when possible, two short sentences max."
#
# Phase 1 of the v3 plan slimmed the prompt so those phrases no longer
# exist in the prompt -- but we keep this detector as defence-in-depth
# in case (a) a stale prompt is in flight, (b) the model hallucinates a
# similar structural template, or (c) any future cloud fallback emits
# the same shape.
_PROMPT_LEAK_FINGERPRINT_RE = re.compile(
    r"""
    ^\s*
    (?:
        # v3-removed FINAL-ANSWER RULES block
        the\s+final\s+answer\s+only\b |
        reply\s+with\s+the\s+final\s+answer\b |
        one\s+short\s+(?:jarvis-style\s+)?line\b |
        plain\s+text\s+only\b |
        # v3-removed RESPONSE RULES block
        if\s+the\s+question\s+is\s+a\s+simple,?\s+short,?\s+or\s+info\s+query\b |
        give\s+one\s+short\s+sentence\s+when\s+possible\b |
        two\s+short\s+sentences\s+max\b |
        # v3-removed VOICE OUTPUT RULES (V1-V9) verbatim leaks
        output\s+only\s+the\s+final\s+answer\b |
        spoken\s*=\s*final\s+answer\b |
        if\s+the\s+thought\s+feels\s+like\s+planning\b |
        boss\s+only\s+hears\s+what'?s\s+spoken\b |
        # Any reply that opens with "respond in plain text" or
        # "no markdown, no bullets" is the model echoing OUTPUT STYLE.
        respond\s+in\s+plain\s+text\b |
        no\s+markdown,?\s+no\s+bullets\b |
        # Catches the v3 STYLE FINGERPRINT being parroted (would be a
        # regression but cheap to guard against).
        output\s+style\s*:\s+spoken\s+plain\s+text\b |
        brevity\s+required\.\s+aim\s+for\s+~?\s*15\s+words\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _looks_like_prompt_leak(text: str) -> bool:
    """Return True when a fragment matches a known prompt-text fingerprint.

    These are imperative phrases that should only ever appear inside the
    system prompt -- never in a spoken reply. If we see them in the LLM
    output, the model has confused the rules for the answer.
    """
    if not text:
        return False
    head = text.strip()
    if not head:
        return False
    return bool(_PROMPT_LEAK_FINGERPRINT_RE.match(head))


MAX_REACT_STEPS = 3


class LocalBrainController:
    """Agentic LLM brain with tool-use and ReAct reasoning loop.

    The LLM sees all available tools in its prompt. When it decides
    an action is needed, it outputs tool_call tags. These are parsed,
    executed through the security-gated ActionExecutor, and the results
    are fed back as observations for the next reasoning step.
    """

    def __init__(
        self,
        bus: "AsyncEventBus",
        prompt_builder: "StructuredPromptBuilder",
        config: dict,
        brain_mode_manager: "BrainModeManager | None" = None,
    ) -> None:
        self._bus = bus
        self._prompt_builder = prompt_builder
        self._config = config
        self._brain_mode_manager = brain_mode_manager

        from brain.mlx_llm import MLXBrain
        self._llm = MLXBrain(config)
        if brain_mode_manager is not None:
            self._llm.set_brain_mode_manager(brain_mode_manager)

        self._action_executor: ActionExecutor | None = None

        self._total_calls = 0
        self._total_tokens_approx = 0
        self._total_tool_calls = 0
        self._total_react_loops = 0
        self._first_token_latencies: list[float] = []
        self._generation_id: int = 0

        self._inference_guard: Any = None
        self._curiosity_engine: Any = None

        self._rag_engine: Any = None
        self._gpu_coord: Any = None
        _rc = config.get("rag") or {}
        self._rag_budget_ms = float(_rc.get("first_token_budget_ms", 120))

        self._timeline: Any = None
        self._mode_resolver: Any = None
        self._prefetch_engine: Any = None
        self._memory_graph: Any = None
        self._second_brain: Any = None
        self._recent_queries: deque[str] = deque(maxlen=12)
        self._current_runtime_mode: str = "SMART"
        self._last_mode_info: Dict[str, Any] = {}
        self._feedback_engine: Any = None
        self._system_monitor: Any = None
        self._suggester: Any = None
        self._runtime_watchdog: Any = None
        self._prev_predictions: list[str] = []
        self._last_retrieval_source: str = ""
        self._v7_context_last: V7RuntimeContext | None = None
        self._latency_board_llm: str = "llm_large"
        self._report_dir = Path("logs/reports")
        self._report_export_min_words = 140
        self._report_export_min_chars = 900
        self._response_language: str = "english"

        # v22: Cloud intelligence components
        self._confidence_engine: Any = None
        self._decision_engine: Any = None
        self._gemini_client: Any = None
        self._semantic_cache: Any = None
        self._preference_store: Any = None
        # Sprint A4: optional on-device VLM captioner (SmolVLM by
        # default). When ``_gemini_client`` is None the PERCEPTION
        # branch falls through to this captioner instead of speaking
        # the legacy "Gemini Client offline" string.
        self._vlm_captioner: Any = None
        self._vision_engine: Any = None

        # Optional vetter — invoked on the final LLM text before it leaves
        # the controller. Wired from the Router so it can apply verb-match
        # and low-confidence guardrails that transform action-promise
        # hallucinations into short clarifying questions.
        self._response_vetter: Any = None

        # Per-turn streaming state used to break the guardrail-cascade. When
        # the stream vetter rewrites a sentence mid-flight we mark the turn
        # as ``_turn_vetter_rewrote`` so the end-of-turn logic can (a) skip
        # a redundant second vet pass and (b) accept the clarifier as the
        # final response instead of triggering strict recovery. The emitted
        # sentence buffer (``_turn_emitted_sentences``) lets recovery paths
        # recover the already-spoken text rather than emit "I lost that".
        self._turn_vetter_rewrote: bool = False
        self._turn_emitted_sentences: list[str] = []

    def attach_response_vetter(self, vetter: Any) -> None:
        """Wire a callable `vetter(query, reply, confidence) -> str` that
        may rewrite the reply to a safer clarification when grounding is
        weak. Set to None to disable."""
        self._response_vetter = vetter

        from core.cognition.intent_classifier import IntentClassifier
        from core.cognition.planner import PlannerEngine
        from core.cognition.state_graph import SystemStateGraph
        
        self._intent_classifier = IntentClassifier()
        self._planner = PlannerEngine(ai_client=self._llm)
        self._state_graph = SystemStateGraph()

    def _prediction_prefetch_enabled(self) -> bool:
        if self._brain_mode_manager is None:
            return True
        try:
            return bool(self._brain_mode_manager.feature_enabled("prediction_prefetch"))
        except Exception:
            return True

    def attach_feedback_engine(self, engine: Any) -> None:
        self._feedback_engine = engine

    def attach_system_monitor(self, monitor: Any) -> None:
        self._system_monitor = monitor

    def attach_suggester(self, suggester: Any) -> None:
        self._suggester = suggester

    def attach_timeline(self, timeline: Any) -> None:
        self._timeline = timeline

    def attach_mode_resolver(self, resolver: Any) -> None:
        self._mode_resolver = resolver

    def attach_prefetch_engine(self, engine: Any) -> None:
        self._prefetch_engine = engine

    def attach_memory_graph(self, graph: Any) -> None:
        self._memory_graph = graph

    def attach_second_brain(self, second_brain: Any) -> None:
        self._second_brain = second_brain

    def attach_vlm_captioner(self, captioner: Any) -> None:
        """Sprint A4: wire the on-device VLM (SmolVLM by default) for
        the PERCEPTION branch fallback. Without this the controller
        used to speak the legacy ``"Gemini Client offline"`` string
        because the captioner stayed in main.py scope."""
        self._vlm_captioner = captioner
        if captioner is not None:
            logger.info(
                "VLM captioner attached to local brain (vision fallback ready)",
            )

    def attach_vision_engine(self, vision_engine: Any) -> None:
        """Wire the camera VisionEngine for PERCEPTION disambiguation.

        The local brain's broad PERCEPTION classifier previously sent all
        such queries to ``ScreenReader.analyze_screen``. If router-level
        intent matching misses a trailing-STT fragment ("can you see me at"),
        this handle lets the brain still route camera-facing phrases to the
        actual camera instead of describing the laptop screen.
        """
        self._vision_engine = vision_engine

    def attach_cloud_intelligence(
        self,
        *,
        confidence_engine: Any = None,
        decision_engine: Any = None,
        gemini_client: Any = None,
        semantic_cache: Any = None,
        preference_store: Any = None,
    ) -> None:
        """v22: Wire cloud intelligence components for post-LLM scoring."""
        self._confidence_engine = confidence_engine
        self._decision_engine = decision_engine
        self._gemini_client = gemini_client
        self._semantic_cache = semantic_cache
        self._preference_store = preference_store
        logger.info(
            "Brain v22 cloud intelligence: confidence=%s, decision=%s, "
            "gemini=%s, sem_cache=%s, prefs=%s",
            confidence_engine is not None, decision_engine is not None,
            gemini_client is not None, semantic_cache is not None,
            preference_store is not None,
        )

    def attach_rag(
        self,
        rag_engine: Any = None,
        gpu_coordinator: Any = None,
    ) -> None:
        """Wire low-latency RAG (optional). ``gpu_coordinator`` for observability snapshot."""
        self._rag_engine = rag_engine
        self._gpu_coord = gpu_coordinator

    @staticmethod
    def _compact_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    @staticmethod
    def _append_offer_to_reply(reply: str, offer_text: str) -> str:
        """Glue a Jarvis offer onto the LLM reply with sane spacing.

        Sprint J: the synthesizer hands us a one-liner like
        ``"Want me to open Chrome for you, Boss?"`` and we splice it
        onto the end of the model's answer. Three small rules so the
        result reads naturally on TTS:

          * If the model already ends mid-thought (no ``.``/``!``/``?``)
            we add a single period before the offer so the prosody
            engine doesn't rush them together.
          * If the LLM somehow already produced a "Want me to..." /
            "Should I..." / "Shall I..." line, we DO NOT double-offer.
            The synthesizer's deterministic offer is dropped from the
            spoken text (the registry stash still happens upstream so
            the next-turn confirm still works).
          * We always insert a single space between the two so the
            engine doesn't elide the question mark.
        """
        body = (reply or "").rstrip()
        offer = (offer_text or "").strip()
        if not offer:
            return body
        if not body:
            return offer
        lower_tail = body[-160:].lower()
        # Heuristic: model already proposed an action -- don't echo.
        for cue in (
            "want me to", "would you like me to", "should i",
            "shall i", "do you want me to",
        ):
            if cue in lower_tail:
                return body
        if body[-1] not in ".!?…":
            body = body + "."
        return f"{body} {offer}"

    def _sanitize_emittable_text(self, text: str) -> str:
        # Defense-in-depth: cut at any leaked ChatML / HF control token and
        # strip stray copies of them. This catches cloud-fallback paths and
        # any stale stream that bypasses the MLX-side stop-sequence check.
        raw = self._compact_text(text)
        # Very first check — BEFORE we strip leading quotes — catch the
        # "\" or similar." family of fragments. These are short dangling
        # quote-tails the LLM leaks after a quoted user-echo; if we strip
        # the leading quote first they become real-looking ("or similar.")
        # and a declarative-sentence detector won't catch them.
        if raw and _LONE_QUOTE_TAIL_RE.match(raw):
            tail_words = len(raw.split())
            if tail_words <= 4:
                logger.warning(
                    "Suppressing lone-quote-tail fragment before TTS: '%s'",
                    raw[:80],
                )
                return ""
        if raw:
            raw = _STRIP_HARD_STOP_TOKENS_RE.split(raw, maxsplit=1)[0]
            raw = _STRIP_SPECIAL_TOKENS_RE.sub("", raw)
            raw = _STRIP_TRANSCRIPT_HEADERS_RE.split(raw, maxsplit=1)[0]
            # Strip a leading lone quote-wrap (mirrors MLX-side guard so
            # cloud-fallback paths and pre-stripped fragments stay clean).
            stripped = _LEADING_QUOTE_WRAP_RE.sub("", raw, count=1)
            if stripped != raw:
                stripped = _TRAILING_UNCLOSED_QUOTE_RE.sub("", stripped).rstrip()
                raw = stripped
            raw = raw.strip()
        cleaned = _INLINE_TRACE_RE.sub("", raw).strip(" -:>")
        if not cleaned:
            return ""

        # Hard reject: the entire fragment is narration about the user
        # (e.g. "the user is greeting you, respond politely and warmly").
        # Small instruction-tuned models still leak this even with strict
        # system rules; if even a single streamed clause matches, we drop
        # the whole emission so TTS never speaks our own instructions.
        if _looks_like_pure_instruction_leak(cleaned):
            logger.warning(
                "Suppressing instruction-leak clause before TTS: '%s'",
                cleaned[:120],
            )
            return ""

        # Peel any leading chain-of-thought preface (defense-in-depth; the
        # MLX guard should have already stripped these, but non-MLX paths
        # and mid-stream flushes still land here with narration intact).
        cleaned = _strip_cot_preface(cleaned)
        if not cleaned:
            return ""

        # Reasoning-narration sentences (no CoT preface, but the whole
        # sentence is the model thinking out loud). These slipped past the
        # strippers above because they're declarative, not prefaces:
        #   - "Based on the response contract, I should confirm my activity."
        #   - "First, looking at the conversation history..."
        #   - "Now they're asking about what happened to me."
        #   - "I need to help the user find ..."
        #   - "I don't need to use any tools here since it's straightforward."
        #   - "Just respond with the standard message."
        #   - "Wait, looking at the available tools, ..."
        #   - "Since I can't browse the internet directly, I should ..."
        #   - "The user previously asked if I was active properly."
        #   - "The answer needs to be concise and in plain text."
        #   - Lone fragments like "\" or similar." — a dangling quote with
        #     only 1-3 words, usually the tail of a quoted user-echo leak.
        if _looks_like_reasoning_leak(cleaned):
            logger.warning(
                "Suppressing reasoning-leak sentence before TTS: '%s'",
                cleaned[:120],
            )
            return ""

        # v3 prompt-text-leak fingerprint check. Catches the case where the
        # LLM regurgitated a literal line from its own system prompt as the
        # spoken answer (see _looks_like_prompt_leak docstring for context).
        if _looks_like_prompt_leak(cleaned):
            logger.warning(
                "Suppressing prompt-leak fragment before TTS: '%s'",
                cleaned[:120],
            )
            return ""

        label_hits = len(_TRANSCRIPT_LABEL_RE.findall(cleaned))
        if label_hits >= 2:
            assistant_segments = [
                self._compact_text(seg).strip(" -:>")
                for seg in re.findall(
                    r"(?:ATOM|Assistant):\s*(.+?)(?=(?:User|Boss|ATOM|Assistant):|$)",
                    cleaned,
                    re.I,
                )
            ]
            assistant_segments = [
                seg for seg in assistant_segments if seg and not _REPEATED_SPEAKER_RE.match(seg)
            ]
            if assistant_segments:
                cleaned = max(assistant_segments, key=len)
            else:
                return ""

        cleaned = _TRANSCRIPT_LABEL_RE.sub("", cleaned).strip(" -:>")
        cleaned = re.sub(
            r"(?:\b(?:User|Boss|ATOM|Assistant)\b\s*)+$",
            "",
            cleaned,
            flags=re.I,
        ).strip(" -:>")
        if not cleaned or _REPEATED_SPEAKER_RE.match(cleaned):
            return ""

        words = cleaned.lower().replace(":", " ").split()
        if len(words) >= 3 and len(set(words)) == 1 and words[0] in {"atom", "user", "assistant", "boss"}:
            return ""
        return cleaned

    def _strip_model_artifacts(self, query: str, text: str) -> str:
        cleaned = self._compact_text(text)
        if not cleaned:
            return ""

        compact_query = self._compact_text(query)
        if compact_query:
            direct_reply = re.search(
                rf"(?:User|Boss):\s*{re.escape(compact_query)}\s*"
                rf"(?:ATOM|Assistant):\s*(.+?)(?=(?:User|Boss|ATOM|Assistant):|$)",
                cleaned,
                re.I,
            )
            if direct_reply:
                cleaned = direct_reply.group(1).strip()

        cleaned = self._sanitize_emittable_text(cleaned)
        if compact_query and cleaned.lower() == compact_query.lower():
            return ""
        return cleaned

    def _finalize_inline_text(self, query: str, text: str) -> str:
        from core import adaptive_personality as personality

        cleaned = self._strip_model_artifacts(query, text)
        if not cleaned:
            cleaned = "I lost the thread there, Boss. Ask that again."
        if classify_response_mode(query) is ResponseMode.SHORT:
            cleaned = summarize_report(cleaned, max_sentences=2, max_chars=180)
        return personality.polish_response(cleaned, source="local_brain")

    @staticmethod
    def _max_tokens_override(
        *,
        response_mode: ResponseMode,
        budget_tier: str,
        requested_tier: str,
    ) -> int | None:
        budget = str(budget_tier or "").strip().lower()
        requested = str(requested_tier or "").strip().lower()
        # v3.3 brain: caps calibrated for Qwen3-4B-Instruct-2507-4bit.
        # Empirically Qwen needs room comparable to Phi-3.5-mini for the
        # same answer quality (live smoke: 60-word reply at max_tokens=64
        # still coherent), so we keep the Phi-era ceilings. Above SHORT
        # ~110 tokens the model starts to narrate instead of answer --
        # caps remain tight on purpose. The SKILL.md invariant
        # "max_tokens <= 320 for voice turns" is enforced transitively
        # here: DETAIL=256, REPORT=unbounded (non-voice only).
        if response_mode is ResponseMode.SHORT or budget in {"command", "info"}:
            return 96
        if budget == "simple":
            return 128
        if response_mode is ResponseMode.DETAIL or budget == "complex" or requested == "complex":
            return 256
        if response_mode is ResponseMode.REPORT or budget == "creative" or requested == "creative":
            return None
        return 160

    @staticmethod
    def _repair_max_tokens_override(max_tokens_override: int | None) -> int:
        if not max_tokens_override:
            return 128
        return max(96, min(int(max_tokens_override), 160))

    def _build_repair_prompt(
        self,
        query: str,
        *,
        context: dict[str, str] | None = None,
    ) -> str:
        safe_query = re.sub(r"\s+", " ", (query or "").strip())
        base_prompt = self._prompt_builder.build(
            safe_query,
            memory_summaries=None,
            history=[],
            context=context,
            document_context=None,
            observations=None,
            rag_enrichment=None,
        )
        return (
            f"{base_prompt}\n\n"
            "FINAL RETRY:\n"
            "Start immediately with the actual answer.\n"
            "Use plain text only.\n"
            "No role labels, no transcript format, no tool calls, no meta instructions.\n"
            f"Question: {safe_query}\n"
            "Answer:"
        )

    def _reject_low_quality_answer(self, query: str, text: str) -> bool:
        # NOTE: Do NOT call ``.strip('"')`` here. The instruction-echo
        # rejector requires the *opening* and *closing* quote characters
        # to remain paired so the leading-quoted-prefix regex can fire on
        # leaks like ``"Yeah Boss" — the user is greeting you, ...``. If
        # we strip just one quote, the prefix breaks and the leak slips
        # straight into TTS as ATOM speaking its own instructions.
        compact = self._compact_text(text).strip()
        if not compact:
            return True
        if _INSTRUCTION_ECHO_RE.search(compact):
            return True
        # Only NOW collapse paired wrapping quotes for the remaining
        # similarity / question / memory checks.
        clean = compact
        if (
            len(clean) >= 2
            and clean[0] in '"\u201c\u2018\u2019\u201d`'
            and clean[-1] in '"\u201c\u2018\u2019\u201d`'
        ):
            clean = clean[1:-1].strip()
        if not clean:
            return True
        if _looks_like_pure_instruction_leak(clean):
            return True
        normalized_query = normalize_query(query)
        normalized_clean = normalize_query(clean)
        if normalized_query and normalized_clean == normalized_query:
            return True
        if _IMPERATIVE_ECHO_RE.search(clean) and any(token in normalized_query for token in ("explain", "compare", "what is", "why", "how")):
            return True
        if clean.endswith("?") and normalized_clean:
            query_words = set(normalized_query.split())
            answer_words = set(normalized_clean.split())
            if query_words and len(query_words & answer_words) >= max(2, min(4, len(query_words))):
                return True
        if _MEMORY_ACK_RE.fullmatch(clean):
            if not any(token in normalized_query for token in ("remember", "recall", "what do you know", "have i told")):
                return True
        return False

    def _candidate_response_text(self, query: str, raw_response: str, parsed: Any) -> str:
        candidate = self._compact_text(getattr(parsed, "text_response", "") or "")
        if not candidate and raw_response and not getattr(parsed, "has_tool_calls", False):
            candidate = self._strip_model_artifacts(query, raw_response)
        if not candidate:
            return ""
        if self._reject_low_quality_answer(query, candidate):
            logger.warning("Rejected low-quality candidate response: %s", candidate[:200])
            self._bus.emit("metrics_event", counter="llm_response_rejected")
            return ""
        return candidate

    def _report_path_for(self, query: str) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        slug = slugify_query(query)
        return self._report_dir / f"{stamp}_{slug}.txt"

    def _write_report_file(self, query: str, text: str) -> Path:
        path = self._report_path_for(query)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            f"ATOM Research Report\n"
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Query: {query.strip()}\n"
            f"{'-' * 72}\n\n"
            f"{text.strip()}\n"
        )
        path.write_text(payload, encoding="utf-8")
        return path

    def _remember_report(self, query: str, summary: str, path: Path) -> None:
        if self._second_brain is None:
            return
        try:
            if hasattr(self._second_brain, "remember_report"):
                self._second_brain.remember_report(
                    topic=query,
                    summary=summary,
                    path=path.as_posix(),
                )
                return
            self._second_brain.learn_fact(
                (
                    f"Saved report for '{query}': {summary} "
                    f"(file: {path.as_posix()})"
                ),
                source="report_export",
                tags=["report", "summary"],
                importance=0.72,
            )
        except Exception:
            logger.info("Failed to remember exported report", exc_info=True)

    def _maybe_export_report(
        self,
        query: str,
        full_text: str,
    ) -> tuple[str, str | None]:
        # Gate 1 — the *raw* response must look long enough.
        if not should_export_report(
            query,
            full_text,
            min_words=self._report_export_min_words,
            min_chars=self._report_export_min_chars,
        ):
            return self._finalize_inline_text(query, full_text), None

        # Gate 2 — the *sanitized* response (after CoT / quote / token strip)
        # must STILL be long enough. This prevents 'I saved the full report'
        # being spoken every time the model leaks a long chain-of-thought
        # block that gets cut down to a few words by the sanitiser.
        sanitized = self._sanitize_emittable_text(full_text)
        if not sanitized:
            return self._finalize_inline_text(query, full_text), None
        word_count = len(sanitized.split())
        char_count = len(sanitized)
        if (
            word_count < self._report_export_min_words
            or char_count < self._report_export_min_chars
        ):
            logger.info(
                "Skipping report export: sanitized=%d words / %d chars "
                "below threshold (raw=%d chars).",
                word_count,
                char_count,
                len(full_text),
            )
            return self._finalize_inline_text(query, full_text), None

        # Gate 3 — sanitiser must not have removed the bulk of the response.
        # If the sanitised body is < 60% of the raw length, the raw was mostly
        # CoT/garbage and not a real long-form answer worth saving as a file.
        if len(full_text) > 0 and (len(sanitized) / float(len(full_text))) < 0.60:
            logger.info(
                "Skipping report export: sanitiser removed %.1f%% of raw response.",
                100.0 * (1.0 - len(sanitized) / float(len(full_text))),
            )
            return self._finalize_inline_text(query, full_text), None

        try:
            path = self._write_report_file(query, sanitized)
            summary = summarize_report(sanitized, max_sentences=2, max_chars=200)
            self._remember_report(query, summary, path)
            self._bus.emit("report_saved", query=query, path=path.as_posix(), summary=summary)
            self._bus.emit("text_display", text=f"Full report saved: {path.as_posix()}")
            if summary:
                return (
                    self._finalize_inline_text(
                        query,
                        f"I saved the full report, Boss. Quick take: {summary}",
                    ),
                    path.as_posix(),
                )
            return (
                self._finalize_inline_text(query, "I saved the full report, Boss."),
                path.as_posix(),
            )
        except Exception:
            logger.exception("Report export failed")
            return self._finalize_inline_text(query, full_text), None

    async def _retry_with_late_rag(
        self,
        text: str,
        memory_context: list[str] | None,
        context: dict[str, str] | None,
        history: list[tuple[str, str]] | None,
        res: Any,
        trace_id: str | None,
        query_plan: Any | None = None,
        late_depth: int = 0,
        scheduled_gen_id: int = 0,
    ) -> None:
        """Single follow-up generation with high-confidence RAG after budget miss."""
        await asyncio.sleep(0.06)
        if scheduled_gen_id and self._generation_id != scheduled_gen_id:
            return
        await self.on_query(
            text,
            memory_context=memory_context,
            context=context,
            history=history or [],
            trace_id=trace_id,
            query_plan=query_plan,
            enriched_rag_result=res,
            _retry_from_late=True,
            _late_depth=late_depth + 1,
        )

    def attach_gpu_resource_manager(self, mgr: Any) -> None:
        """Legacy shim — use attach_inference_guard instead."""
        self._inference_guard = mgr

    def attach_inference_guard(self, guard: Any) -> None:
        self._inference_guard = guard

    def attach_curiosity_engine(self, engine: Any) -> None:
        self._curiosity_engine = engine

    def attach_runtime_watchdog(self, watchdog: Any) -> None:
        self._runtime_watchdog = watchdog

    def apply_memory_pressure(self, memory_pct: float) -> None:
        if self._rag_engine is not None:
            try:
                self._rag_engine.apply_memory_pressure(memory_pct)
            except Exception:
                logger.info("Local brain RAG pressure hook failed", exc_info=True)
        if self._memory_graph is not None:
            try:
                self._memory_graph.apply_memory_pressure(memory_pct)
            except Exception:
                logger.info("Local brain MemoryGraph pressure hook failed", exc_info=True)

    def drop_prompt_caches(self, reason: str = "pressure") -> None:
        """Forward the KV-cache drop request to the underlying MLX brain.
        Falls back silently when the active backend (e.g. GGUF fallback)
        has no prompt cache implementation.
        """
        llm = self._llm
        fn = getattr(llm, "drop_prompt_caches", None)
        if callable(fn):
            try:
                fn(reason=reason)
            except Exception:
                logger.info("Local brain prompt-cache drop failed", exc_info=True)

    def get_perf_snapshot(self) -> dict[str, Any]:
        """Forward the brain's lifetime perf snapshot for periodic logging.

        Returns ``{}`` for backends without ``get_perf_snapshot`` (e.g. the
        legacy GGUF path) so callers can use ``snap or None`` and skip the
        log line cleanly.
        """
        llm = self._llm
        fn = getattr(llm, "get_perf_snapshot", None)
        if callable(fn):
            try:
                snap = fn()
                if isinstance(snap, dict):
                    return snap
            except Exception:
                logger.debug("Perf snapshot fetch failed", exc_info=True)
        return {}

    def set_thermal_clamp(self, ratio: float, *, reason: str = "") -> None:
        """Forward a thermal ``max_tokens`` multiplier to the backend.

        Ignored silently for backends without a thermal hook.
        """
        llm = self._llm
        fn = getattr(llm, "set_thermal_clamp", None)
        if callable(fn):
            try:
                fn(ratio, reason=reason)
            except Exception:
                logger.info("Local brain thermal clamp failed", exc_info=True)

    def set_action_executor(self, executor: "ActionExecutor") -> None:
        """Inject the ActionExecutor after Router initialization.

        Idempotent: re-injecting the same executor is a no-op and does
        not re-log. ``main.py`` legitimately calls this twice -- once
        right after the controller is built and once after the tool
        registry is populated -- but the second call passes the same
        instance, so we only log on the first attach.
        """
        if self._action_executor is executor:
            return
        self._action_executor = executor
        logger.info("Action executor connected to brain controller")

    @property
    def available(self) -> bool:
        return self._llm.is_available()

    @property
    def is_loaded(self) -> bool:
        return self._llm.is_loaded

    def request_preempt(self) -> None:
        self._llm.request_abort_preempt()

    def is_mlx_generating(self) -> bool:
        """True while the MLX worker is in a generate/stream call."""
        try:
            return bool(self._llm.is_generating())
        except Exception:
            return False

    def unload_llm_for_power(self) -> None:
        """V7: release model memory (next query will reload)."""
        self._llm.shutdown()

    def request_profile_demote(self, *, reason: str = "watchdog") -> bool:
        """Drop the brain to a lighter profile after repeated timeouts.

        Called by :class:`core.runtime_watchdog.RuntimeWatchdog` once the
        consecutive-timeout threshold trips. We force ``runtime_mode=FAST``
        (skips late-RAG, heavy memory-graph prefetch, reflection loops) AND
        unload the model so the next turn rehydrates a clean session. This is
        idempotent: the controller flips only if it's not already demoted.
        """
        already_fast = str(self._current_runtime_mode or "").upper() == "FAST"
        if already_fast:
            logger.debug(
                "request_profile_demote skipped -- brain already in FAST (reason=%s)",
                reason,
            )
            return False
        try:
            self._current_runtime_mode = "FAST"
            self._last_mode_info = {
                "runtime_mode": "FAST",
                "reason": f"watchdog_demote:{reason}",
                "source": "runtime_watchdog",
            }
        except Exception:
            logger.debug(
                "request_profile_demote could not stamp mode info",
                exc_info=True,
            )
        try:
            self._llm.shutdown()
        except Exception:
            logger.debug(
                "request_profile_demote LLM shutdown failed",
                exc_info=True,
            )
        logger.warning(
            "LocalBrainController: runtime mode demoted to FAST (reason=%s). "
            "RAG/reflection skipped until operator flips back to SMART.",
            reason,
        )
        return True

    async def warm_up(
        self,
        *,
        model_role: str | None = None,
        load_all: bool = False,
    ) -> bool:
        if not self._llm.is_available():
            logger.warning(
                "Local brain not available "
                "(MLX model directory missing or mlx/mlx_lm not installed)",
            )
            return False
        role_label = "all_roles" if load_all else (model_role or "default_role")
        logger.info("Local brain: warming up MLX model (%s)...", role_label)
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        loaded = await loop.run_in_executor(
            None,
            partial(self._llm.preload, model_role=model_role, load_all=load_all),
        )
        elapsed = (time.monotonic() - t0) * 1000
        if loaded:
            logger.info(
                "Local brain ready in %.0fms (MLX, role=%s, agentic mode)",
                elapsed,
                role_label,
            )
        else:
            logger.warning("Local brain warm-up failed (role=%s)", role_label)
        return bool(loaded)

    async def _handle_query_failure(self, source: str, exc: Exception) -> None:
        logger.exception("Local brain %s failed: %s", source, exc)
        try:
            self._bus.emit("metrics_event", counter="errors_total")
        except Exception:
            logger.debug('Metrics counter emit failed', exc_info=True)
        try:
            self._bus.emit("llm_error", source="local", error=str(exc)[:200])
        except Exception:
            logger.debug('Metrics counter emit failed', exc_info=True)
        try:
            self._bus.emit_long(
                "response_ready",
                text="Local brain hit an error, Boss. Check the log and try again.",
            )
        except Exception:
            logger.info("Local brain fallback response failed", exc_info=True)

    async def on_query(
        self,
        text: str,
        memory_context: list[str] | None = None,
        context: dict[str, str] | None = None,
        history: list[tuple[str, str]] | None = None,
        **_kw: Any,
    ) -> None:
        try:
            await self._on_query_impl(
                text,
                memory_context=memory_context,
                context=context,
                history=history,
                **_kw,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._handle_query_failure("on_query", exc)

    async def _on_query_impl(
        self,
        text: str,
        memory_context: list[str] | None = None,
        context: dict[str, str] | None = None,
        history: list[tuple[str, str]] | None = None,
        **_kw: Any,
    ) -> None:
        """Process a query through the agentic LLM with ReAct loop.

        1. Build prompt with tools visible
        2. LLM generates response (streamed)
        3. Parse response for tool calls
        4. If tools called: execute, collect observations, re-prompt LLM
        5. Repeat up to MAX_REACT_STEPS
        6. Emit final text response for TTS
        """
        if not self._llm.is_available():
            self._bus.emit_long(
                "response_ready",
                text=(
                    "Local brain is not available. Check that the MLX model "
                    "directories exist and that mlx/mlx_lm are installed."
                ),
            )
            return

        # Belt-and-braces: in case any caller ever splices a "[SYSTEM NOTE: …]"
        # block back into the user query (older code paths used to do this
        # for repeat queries before we moved the steer to a system-layer
        # ``repeat_hint``), strip it here so the LLM never sees it. If we
        # leak it, small instruction-tuned models echo the bracketed text
        # back during TTS as quoted analysis ("Dear Boss" — the user is …).
        if text and "[SYSTEM NOTE" in text:
            text = re.sub(r"\[SYSTEM NOTE:[^\]]*\](?:[^\n]*\n?)*", "", text).strip()

        self._generation_id += 1
        gen_id = self._generation_id
        # Reset per-turn streaming cascade-break state (see docstring on
        # self._turn_vetter_rewrote in __init__).
        self._turn_vetter_rewrote = False
        self._turn_emitted_sentences = []

        from context.privacy_filter import redact as _redact
        logger.info("Agentic brain query: '%s'", _redact(text[:120]))
        
        # ── Jarvis Convergence: Cognitive Pre-processing ──
        import time
        self._state_graph.last_query_time = time.time()
        intent = self._intent_classifier.classify(text)
        logger.info("Cognitive Intent detected: %s (conf: %.2f)", intent.category.name, intent.confidence)
        
        if intent.category.name == "PERCEPTION":
            try:
                from core.intent_engine import vision_intents

                native_vision = vision_intents.check(text)
            except Exception:
                native_vision = None
            if (
                native_vision is not None
                and getattr(native_vision, "action", "") == "vision_describe"
                and self._vision_engine is not None
            ):
                logger.info(
                    "PERCEPTION routed to camera via vision_intents (%s)",
                    getattr(native_vision, "intent", ""),
                )
                prompt = (
                    (getattr(native_vision, "action_args", {}) or {}).get("prompt")
                    or "user-facing self-check"
                )
                try:
                    result = self._vision_engine.look(
                        reason=f"local_brain_perception:{str(prompt)[:40]}",
                        detect_faces=True,
                        detect_barcodes=False,
                        describe=True,
                    )
                    if getattr(result, "description", ""):
                        self._bus.emit_long(
                            "response_ready",
                            text=str(result.description).strip(),
                        )
                    elif getattr(result, "ok", False):
                        faces = int(getattr(result, "faces", 0) or 0)
                        if faces > 0:
                            text_out = f"I can see you, Boss. {faces} face detected."
                        else:
                            text_out = "Camera is on, Boss, but I don't see a face clearly."
                        self._bus.emit_long("response_ready", text=text_out)
                    else:
                        self._bus.emit_long(
                            "response_ready",
                            text="Camera describe failed, Boss — see logs for details.",
                        )
                    return
                except Exception:
                    logger.warning("PERCEPTION camera route failed", exc_info=True)
                    self._bus.emit_long(
                        "response_ready",
                        text="Camera describe failed, Boss — see logs for details.",
                    )
                    return

            logger.info("PERCEPTION isolated: Bypassing inference loop for Vision AI.")
            from core.perception.screen_reader import ScreenReader
            vision = ScreenReader(
                gemini_client=self._gemini_client,
                vlm_captioner=self._vlm_captioner,
            )
            result = await vision.analyze_screen(text)
            if self._response_vetter is not None and isinstance(result, str) and result:
                try:
                    vetted = self._response_vetter(text, result, 0.6)
                    if isinstance(vetted, str) and vetted.strip() and vetted != result:
                        logger.info(
                            "Router guardrail rewrote PERCEPTION reply: '%s' -> '%s'",
                            result[:80], vetted[:80],
                        )
                        result = vetted
                except Exception:
                    logger.debug("PERCEPTION vetter failed", exc_info=True)
            self._bus.emit_long("response_ready", text=result)
            return
            
        if intent.category.name == "REASONING":
            logger.info("REASONING isolated: Building topological plan graph.")
            plan = await self._planner.generate_plan(text, "")
            if plan.steps:
                plan_text = "MACRO EXECUTION PLAN:\n" + "\n".join([f"- Step {s.step_num}: {s.description} (use tool: {s.target_tool})" for s in plan.steps])
                if context is None:
                    context = {}
                context["planner_directive"] = plan_text
                text = f"You are executing a macro plan. Please follow these steps exactly:\n{plan_text}\n\nUser request: {text}"
        # ──────────────────────────────────────────────────

        trace_id = _kw.get("trace_id")
        if self._inference_guard is not None:
            self._inference_guard.refresh_vram()

        if self._feedback_engine is not None and self._prev_predictions:
            try:
                self._feedback_engine.evaluate_actual_vs_predictions(text, self._prev_predictions)
            except Exception:
                logger.debug('Prediction feedback evaluation failed', exc_info=True)

        policy_query = str(_kw.get("policy_query") or text)
        repeat_hint = bool(_kw.get("repeat_hint", False))

        query_plan = _kw.get("query_plan")
        plan_mode = str(getattr(query_plan, "runtime_mode", "") or "").upper()
        if plan_mode not in {"FAST", "SMART", "DEEP", "SECURE"}:
            plan_mode = ""
        plan_model_role = str(getattr(query_plan, "model_role", "") or "").strip().lower()
        if plan_model_role not in {"fast", "primary"}:
            plan_model_role = None
        self._latency_board_llm = "llm_small" if plan_model_role == "fast" else "llm_large"
        plan_use_rag = getattr(query_plan, "use_rag", None) if query_plan is not None else None
        plan_use_memory = getattr(query_plan, "use_memory", None) if query_plan is not None else None
        plan_prompt_hint = str(getattr(query_plan, "prompt_hint", "") or "").strip()
        plan_reduce_context = bool(getattr(query_plan, "reduce_context", False)) if query_plan is not None else False
        plan_memory_limit = int(getattr(query_plan, "memory_limit", 0) or 0) if query_plan is not None else 0
        plan_history_turn_limit = int(getattr(query_plan, "history_turn_limit", 0) or 0) if query_plan is not None else 0
        plan_rag_budget_ms = float(getattr(query_plan, "rag_budget_ms", 0.0) or 0.0) if query_plan is not None else 0.0
        plan_budget_tier = str(getattr(query_plan, "budget_tier", "") or "").strip().lower() if query_plan is not None else ""
        plan_requested_tier = str(getattr(query_plan, "requested_tier", "") or "").strip().lower() if query_plan is not None else ""
        plan_base_budget_ms = float(getattr(query_plan, "base_budget_ms", 0.0) or 0.0) if query_plan is not None else 0.0
        response_mode = classify_response_mode(policy_query)
        max_tokens_override = self._max_tokens_override(
            response_mode=response_mode,
            budget_tier=plan_budget_tier,
            requested_tier=plan_requested_tier,
        )
        repair_tokens_override = self._repair_max_tokens_override(max_tokens_override)
        should_buffer_response = response_mode in {
            ResponseMode.SHORT,
            ResponseMode.DETAIL,
            ResponseMode.REPORT,
        }
        repair_attempted = False

        mode_override = _kw.get("runtime_mode_override") or (plan_mode or None)
        if query_plan is not None:
            try:
                logger.info(
                    "Local brain plan: tier=%s requested=%s path=%s role=%s mode=%s rag=%s think=%s base=%.0fms budget=%.0fms rag_budget=%.0fms reduce_context=%s reason=%s",
                    plan_budget_tier or "?",
                    plan_requested_tier or "?",
                    getattr(getattr(query_plan, "path", None), "value", getattr(query_plan, "path", "")),
                    plan_model_role or "primary",
                    plan_mode or "SMART",
                    getattr(query_plan, "use_rag", False),
                    getattr(query_plan, "thinking", False),
                    plan_base_budget_ms,
                    float(getattr(query_plan, "budget_ms", 0.0) or 0.0),
                    plan_rag_budget_ms,
                    plan_reduce_context,
                    getattr(query_plan, "reason", ""),
                )
            except Exception:
                logger.info("Local brain query_plan logging failed", exc_info=True)

        gpu_util = 0.0
        if self._gpu_coord is not None:
            try:
                obs = self._gpu_coord.get_observability()
                gs = obs.get("gpu_state") or {}
                gpu_util = float(gs.get("gpu_util_pct", 0) or 0)
            except Exception:
                gpu_util = 0.0

        system_state: dict[str, Any] = {}
        if self._system_monitor is not None:
            try:
                system_state = self._system_monitor.get_system_state()
            except Exception:
                system_state = {}

        user_activity = "active"
        if self._timeline is not None:
            try:
                user_activity = (
                    "idle" if not self._timeline.user_recently_active(120.0) else "active"
                )
            except Exception:
                user_activity = "active"

        pred_acc: float | None = None
        feedback_metrics: dict[str, Any] = {}
        if self._feedback_engine is not None:
            try:
                feedback_metrics = self._feedback_engine.compute_accuracy_metrics()
                pred_acc = float(feedback_metrics.get("prediction_accuracy", 0.5))
            except Exception:
                pred_acc = None

        timeline_summary = ""
        if self._timeline is not None:
            try:
                timeline_summary = self._timeline.summary_for_prompt(
                    window_sec=600.0, max_lines=3,
                )
            except Exception:
                timeline_summary = ""

        mode_ctx = V7RuntimeContext(
            system_state=dict(system_state),
            feedback_metrics=dict(feedback_metrics),
            runtime_mode=self._current_runtime_mode,
            mode_info={},
            timeline_summary=timeline_summary,
            gpu_util_pct=gpu_util,
            prediction_accuracy=pred_acc,
            last_retrieval_source=self._last_retrieval_source,
        )

        if self._mode_resolver is not None:
            self._current_runtime_mode, self._last_mode_info = self._mode_resolver.resolve(
                text,
                gpu_util_pct=gpu_util,
                user_override=mode_override,
                system_state=system_state,
                user_activity=user_activity,
                prediction_accuracy=pred_acc,
                context=mode_ctx,
            )
        else:
            self._current_runtime_mode = mode_override or "SMART"
            self._last_mode_info = {"reason": "query_plan"} if plan_mode else {}

        v7_ctx = mode_ctx.with_mode(self._current_runtime_mode, self._last_mode_info)
        self._v7_context_last = v7_ctx

        if self._timeline is not None:
            try:
                self._timeline.append_event(
                    "llm_query",
                    {"text": text[:2000], "runtime_mode": self._current_runtime_mode},
                )
            except Exception:
                logger.debug('Timeline query event append failed', exc_info=True)
        self._recent_queries.append(text.strip())

        t0_total = time.perf_counter()
        observations: list[str] = []
        all_tool_results: list[str] = []
        text_response_parts: list[str] = []
        react_step = 0
        tool_depth = 0
        MAX_TOOL_DEPTH = 3

        prompt_memory_context = memory_context if plan_use_memory is not False else None
        if prompt_memory_context and plan_memory_limit > 0:
            prompt_memory_context = list(prompt_memory_context[:plan_memory_limit])
        prompt_history = list(history or [])
        if plan_history_turn_limit > 0:
            prompt_history = prompt_history[-plan_history_turn_limit:]
        prompt_context: dict[str, str] | None = context
        response_language = detect_response_language(policy_query, previous=self._response_language)
        self._response_language = response_language
        if response_language:
            prompt_context = dict(prompt_context or {})
            prompt_context["response_language"] = response_language
        if plan_prompt_hint:
            prompt_context = dict(prompt_context or {})
            prompt_context["llm_routing_hint"] = plan_prompt_hint

        rag_document_context: list[str] | None = None
        rag_enrichment: str | None = None
        enriched_in = _kw.get("enriched_rag_result")
        if enriched_in is not None:
            rag_document_context = enriched_in.document_context
            rag_enrichment = enriched_in.enrichment_block or None
        elif (
            self._rag_engine is not None
            and not _kw.get("_retry_from_late")
            and plan_use_rag is not False
        ):
            try:
                from core.rag.query_classifier import classify_query
                from core.rag.rag_engine import RagEngine, retrieve_with_time_budget

                gpu_snap: dict | None = None
                vram_p = 0.0
                if self._gpu_coord is not None:
                    try:
                        obs = self._gpu_coord.get_observability()
                        gpu_snap = obs.get("gpu_state") or {}
                        vram_p = float(gpu_snap.get("fragmentation_estimate", 0) or 0)
                    except Exception:
                        gpu_snap = None

                cx = classify_query(text)
                prefetch_hit_guess = False
                try:
                    if self._rag_engine is not None:
                        prefetch_hit_guess = self._rag_engine._caches.get_retrieval(text) is not None  # noqa: SLF001
                except Exception:
                    prefetch_hit_guess = False
                budget_ms = RagEngine.compute_budget_ms(
                    self._config,
                    cx,
                    gpu_util_pct=gpu_util,
                    vram_pressure=vram_p,
                    prefetch_hit=prefetch_hit_guess,
                )
                if self._runtime_watchdog is not None:
                    budget_ms = self._runtime_watchdog.cap_budget_ms(
                        "rag_retrieval",
                        budget_ms,
                    )
                if plan_rag_budget_ms > 0:
                    budget_ms = min(budget_ms, plan_rag_budget_ms)
                if self._current_runtime_mode == "SECURE":
                    budget_ms *= float(
                        (self._config.get("v7_intelligence") or {}).get(
                            "secure_rag_budget_factor", 0.75,
                        ),
                    )
                if plan_reduce_context:
                    budget_ms *= 0.85
                late_thr = float(
                    (self._config.get("rag") or {}).get("late_restart_confidence", 0.82),
                )
                late_depth = int(_kw.get("_late_depth", 0))
                max_pre = int(
                    ((self._config.get("v7_intelligence") or {}).get("preemption") or {}).get(
                        "max_preemptions_per_query",
                        2,
                    ),
                )

                def _late_rag(res: Any) -> None:
                    from core.cognition.preemption import should_preempt_for_late_rag

                    if late_depth >= max_pre:
                        logger.info(
                            "v7_preemption_blocked reason=max_retries depth=%d max=%d",
                            late_depth,
                            max_pre,
                        )
                        return

                    self._bus.emit_fast(
                        "rag_context_ready",
                        chunks=len(res.chunks),
                        latency_ms=res.latency_ms,
                        confidence=getattr(res, "confidence", 0.0),
                        trace_id=trace_id,
                    )
                    if getattr(res, "confidence", 0) < late_thr or len(res.chunks) < 2:
                        return
                    if not should_preempt_for_late_rag(
                        res,
                        baseline_confidence=0.0,
                        config=self._config,
                    ):
                        return
                    self._bus.emit_fast(
                        "rag_late_high_confidence",
                        confidence=res.confidence,
                        trace_id=trace_id,
                    )
                    if self._generation_id != gen_id:
                        return
                    self.request_preempt()
                    asyncio.create_task(
                        self._retry_with_late_rag(
                            text,
                            memory_context,
                            context,
                            history,
                            res,
                            trace_id,
                            query_plan=query_plan,
                            late_depth=late_depth,
                            scheduled_gen_id=gen_id,
                        ),
                        name="atom_rag_late_restart",
                    )

                late_cb = None if self._current_runtime_mode == "SECURE" else _late_rag

                rag_res = await retrieve_with_time_budget(
                    self._rag_engine,
                    text,
                    budget_ms,
                    memory_summaries=memory_context,
                    system_state=None,
                    gpu_snapshot=gpu_snap,
                    runtime_mode=self._current_runtime_mode,
                    on_late_result=late_cb,
                )
                try:
                    logger.info(
                        "v7_rag_retrieval mode=%s prefetch_guess=%s chunks=%d source=%s",
                        self._current_runtime_mode,
                        prefetch_hit_guess,
                        len(rag_res.chunks),
                        getattr(rag_res, "retrieval_source", ""),
                    )
                except Exception:
                    logger.debug('RAG retrieval log formatting failed', exc_info=True)
                self._last_retrieval_source = str(
                    getattr(rag_res, "retrieval_source", "") or "",
                )
                if self._feedback_engine is not None:
                    try:
                        if getattr(rag_res, "prefetch_hit", False):
                            self._feedback_engine.record_prefetch_event(True)
                        else:
                            self._feedback_engine.record_prefetch_event(False)
                    except Exception:
                        logger.debug('RAG retrieval log formatting failed', exc_info=True)
                try:
                    from core.observability.per_module_latency import get_latency_board

                    rms = float(getattr(rag_res, "latency_ms", 0.0) or 0.0)
                    if rms > 0:
                        get_latency_board().record_module_call("rag", rms, error=False)
                except Exception:
                    logger.debug('RAG retrieval log formatting failed', exc_info=True)
                if rag_res.chunks:
                    rag_document_context = rag_res.document_context
                    rag_enrichment = rag_res.enrichment_block or None
            except Exception:
                logger.warning("RAG retrieve skipped", exc_info=True)

        while react_step <= MAX_REACT_STEPS:
            prompt = self._prompt_builder.build(
                text,
                memory_summaries=prompt_memory_context,
                history=prompt_history,
                context=prompt_context,
                document_context=rag_document_context if react_step == 0 else None,
                observations=observations if observations else None,
                rag_enrichment=rag_enrichment if react_step == 0 else None,
                repeat_hint=repeat_hint and react_step == 0,
            )

            raw_response, first_token_ms, preempted = await self._run_llm_streaming(
                prompt,
                t0_total,
                emit_partial=(react_step == 0 and not observations and not should_buffer_response),
                model_role=plan_model_role,
                max_tokens_override=max_tokens_override,
                policy_query=policy_query,
            )

            if preempted:
                logger.info("Brain preempted at step %d (%.0fms)",
                            react_step, (time.perf_counter() - t0_total) * 1000)
                self._bus.emit("metrics_event", counter="llm_preempted")
                return

            # Cascade break: if the stream vetter already delivered a safe
            # clarifier (or any rewritten sentence) to TTS, treat that as the
            # definitive reply. Triggering strict-recovery here would speak
            # "I lost that answer" on top of the clarifier we just said.
            vetter_delivered = self._turn_vetter_rewrote and bool(
                self._turn_emitted_sentences
            )

            if not raw_response:
                if vetter_delivered:
                    # Adopt the clarifier as final text — do NOT retry.
                    rescued = " ".join(self._turn_emitted_sentences).strip()
                    if rescued:
                        logger.info(
                            "Cascade break: stream vetter already delivered reply "
                            "(%d sentences), skipping strict recovery",
                            len(self._turn_emitted_sentences),
                        )
                        text_response_parts.append(rescued)
                        break
                if react_step == 0 and not observations and not repair_attempted:
                    repair_attempted = True
                    logger.warning("Brain produced no usable text; retrying in strict recovery mode")
                    self._bus.emit("metrics_event", counter="llm_repair_retry")
                    repair_prompt = self._build_repair_prompt(
                        text,
                        context=prompt_context,
                    )
                    raw_response, retry_first_token_ms, preempted = await self._run_llm_streaming(
                        repair_prompt,
                        t0_total,
                        emit_partial=False,
                        model_role=plan_model_role,
                        max_tokens_override=repair_tokens_override,
                        policy_query=policy_query,
                        extra_stop_sequences_override=(),
                    )
                    if retry_first_token_ms and not first_token_ms:
                        first_token_ms = retry_first_token_ms
                    if preempted:
                        logger.info("Brain preempted during strict recovery retry")
                        self._bus.emit("metrics_event", counter="llm_preempted")
                        return
                    if raw_response:
                        logger.info("Strict recovery retry produced answer text")
                    else:
                        logger.warning("Strict recovery retry also returned no usable text")
                break

            parsed = parse_tool_calls(raw_response)
            candidate_text = self._candidate_response_text(policy_query, raw_response, parsed)

            if candidate_text:
                text_response_parts.append(candidate_text)
            elif vetter_delivered:
                # Same cascade-break: candidate was rejected but we already
                # spoke a clarifier. Don't strict-recover; use the clarifier.
                rescued = " ".join(self._turn_emitted_sentences).strip()
                if rescued:
                    logger.info(
                        "Cascade break: candidate rejected but stream vetter "
                        "delivered clarifier — accepting clarifier as final reply",
                    )
                    text_response_parts.append(rescued)
                    break
            elif not parsed.has_tool_calls and react_step == 0 and not observations and not repair_attempted:
                repair_attempted = True
                logger.warning("Brain produced low-quality text; retrying in strict recovery mode")
                self._bus.emit("metrics_event", counter="llm_repair_retry")
                repair_prompt = self._build_repair_prompt(
                    text,
                    context=prompt_context,
                )
                raw_response, retry_first_token_ms, preempted = await self._run_llm_streaming(
                    repair_prompt,
                    t0_total,
                    emit_partial=False,
                    model_role=plan_model_role,
                    max_tokens_override=repair_tokens_override,
                    policy_query=policy_query,
                    extra_stop_sequences_override=(),
                )
                if retry_first_token_ms and not first_token_ms:
                    first_token_ms = retry_first_token_ms
                if preempted:
                    logger.info("Brain preempted during strict recovery retry")
                    self._bus.emit("metrics_event", counter="llm_preempted")
                    return
                parsed = parse_tool_calls(raw_response) if raw_response else parse_tool_calls("")
                candidate_text = self._candidate_response_text(policy_query, raw_response, parsed)
                if candidate_text:
                    logger.info("Strict recovery retry produced valid answer text")
                    text_response_parts.append(candidate_text)
                elif raw_response:
                    logger.warning("Strict recovery retry returned text, but it was rejected")

            if not parsed.has_tool_calls or self._action_executor is None:
                break
                
            tool_depth += 1
            if tool_depth > MAX_TOOL_DEPTH:
                logger.warning("MAX_TOOL_DEPTH (%d) exceeded, breaking ReAct loop", MAX_TOOL_DEPTH)
                text_response_parts.append("I've hit my internal limit for tool calls on this task, Boss. I'm stopping here to prevent a loop.")
                break

            react_step += 1
            self._total_react_loops += 1
            logger.info("ReAct step %d: %d tool call(s)",
                        react_step, len(parsed.tool_calls))

            step_observations: list[str] = []
            for tc in parsed.tool_calls:
                if self._runtime_watchdog is not None:
                    from core.reasoning.action_executor import ActionResult

                    tool_result = await self._runtime_watchdog.run_async(
                        "tool_execution",
                        self._action_executor.execute_async(tc),
                        default=ActionResult(
                            tool_name=tc.name,
                            success=False,
                            error="Tool execution timed out.",
                        ),
                        metadata={"tool": tc.name},
                    )
                    result = tool_result.value
                else:
                    result = await self._action_executor.execute_async(tc)
                self._total_tool_calls += 1
                step_observations.append(result.observation)
                all_tool_results.append(result.observation)

                try:
                    from core.observability.per_module_latency import get_latency_board

                    em = float(getattr(result, "elapsed_ms", 0.0) or 0.0)
                    get_latency_board().record_module_call(
                        "tool_executor",
                        em if em > 0 else 0.01,
                        error=not result.success,
                    )
                except Exception:
                    logger.debug('Tool latency board record failed', exc_info=True)

                self._bus.emit(
                    "tool_executed",
                    tool=tc.name,
                    success=result.success,
                    elapsed_ms=result.elapsed_ms,
                    arguments=dict(tc.arguments or {}),
                )

                if result.needs_confirmation:
                    confirm_text = (
                        f"{parsed.text_response} " if parsed.text_response else ""
                    ) + result.confirmation_prompt
                    self._bus.emit_long("response_ready", text=confirm_text)

                    self._bus.emit("pending_tool_confirmation",
                                   tool_call=tc, result=result)
                    self._emit_final_metrics(
                        t0_total, first_token_ms,
                        " ".join(text_response_parts),
                        trace_id=trace_id,
                    )
                    return

            observations.extend(step_observations)

            if react_step >= MAX_REACT_STEPS:
                logger.info("ReAct loop hit max steps (%d)", MAX_REACT_STEPS)
                break

        elapsed_total = (time.perf_counter() - t0_total) * 1000

        full_text = " ".join(text_response_parts).strip()

        if not full_text and all_tool_results:
            success_results = [r for r in all_tool_results if r.startswith("[OK]")]
            if success_results:
                full_text = ". ".join(
                    r.replace("[OK] ", "").split(": ", 1)[-1]
                    for r in success_results
                )

        if not full_text:
            # Cascade-safe recovery. Previously we always emitted
            # "I lost that answer" + fired llm_error — but if the stream
            # vetter already delivered a clarifier to TTS, that would
            # speak on top of it. Treat three cases explicitly:
            #   1. Vetter delivered clarifier -> reuse it (no llm_error).
            #   2. WH / definition question   -> warm clarifier, no error.
            #   3. Everything else            -> original behavior.
            if self._turn_vetter_rewrote and self._turn_emitted_sentences:
                rescued = " ".join(self._turn_emitted_sentences).strip()
                logger.info(
                    "Empty final text but stream vetter delivered reply — "
                    "reusing clarifier (no llm_error): '%s'", rescued[:120],
                )
                full_text = rescued
            else:
                logger.warning("Brain returned empty response (%.0fms)", elapsed_total)
                ql = (policy_query or "").lower().strip()
                wh_prefixes = (
                    "what ", "what's ", "whats ", "who ", "who's ",
                    "when ", "where ", "why ", "how ", "which ",
                    "define ", "explain ", "tell me about ", "meaning of ",
                    "describe ",
                )
                is_wh = any(ql.startswith(p) for p in wh_prefixes)
                if is_wh:
                    import random
                    wh_clarifiers = (
                        "I need a second on that one, Boss — say it once more?",
                        "Let me catch that again, Boss — mind repeating the question?",
                        "I'm not sure I heard the full question, Boss. One more time?",
                    )
                    recovery_text = random.choice(wh_clarifiers)
                    logger.info(
                        "Empty LLM reply on WH question — emitting warm "
                        "clarifier (no llm_error): '%s'", recovery_text,
                    )
                    self._bus.emit_long("response_ready", text=recovery_text)
                    return
                self._bus.emit(
                    "text_display",
                    text="Recovery note: the local model returned no usable answer for this request.",
                )
                self._bus.emit_long(
                    "response_ready",
                    text="I lost that answer, Boss. Ask it once more.",
                )
                self._bus.emit("llm_error", source="local", error="empty_response")
                return

        # ── v22: Post-LLM Confidence Scoring + Cloud Fallback ────────
        if self._confidence_engine is not None:
            try:
                conf_score = self._confidence_engine.score(policy_query, full_text)
                logger.info(
                    "v22 confidence: %.3f for '%s' (%d chars)",
                    conf_score, policy_query[:50], len(full_text),
                )

                # If confidence is low and Gemini is available → escalate
                if (
                    self._confidence_engine.should_escalate(conf_score)
                    and self._gemini_client is not None
                    and hasattr(self._gemini_client, "is_available")
                    and self._gemini_client.is_available
                ):
                    logger.info(
                        "v22 escalation: confidence=%.3f < threshold, trying Gemini",
                        conf_score,
                    )
                    self._bus.emit_fast("metrics_event", counter="cloud_escalation_attempts")
                    try:
                        cloud_resp, cloud_ok = await self._gemini_client.ask(policy_query)
                        if cloud_ok and cloud_resp and len(cloud_resp) > 20:
                            cloud_score = self._confidence_engine.score(
                                policy_query, cloud_resp,
                            )
                            if cloud_score > conf_score:
                                logger.info(
                                    "v22 escalation SUCCESS: cloud=%.3f > local=%.3f",
                                    cloud_score, conf_score,
                                )
                                full_text = cloud_resp
                                self._bus.emit_fast(
                                    "metrics_event",
                                    counter="cloud_escalation_success",
                                )
                            else:
                                logger.info(
                                    "v22 escalation KEPT LOCAL: cloud=%.3f <= local=%.3f",
                                    cloud_score, conf_score,
                                )
                    except Exception:
                        logger.info("v22 cloud escalation failed", exc_info=True)
            except Exception:
                logger.info("v22 confidence scoring failed", exc_info=True)

        if self._curiosity_engine is not None and text:
            try:
                should_record = repair_attempted
                if not should_record and self._confidence_engine is not None:
                    try:
                        cs = self._confidence_engine.score(policy_query, full_text)
                        should_record = cs < 0.3
                    except Exception:
                        pass
                if should_record:
                    self._curiosity_engine.record_knowledge_gap(text)
            except Exception:
                pass

        # ── v22: Decision Engine enrichment ───────────────────────────
        if self._decision_engine is not None:
            try:
                enriched = self._decision_engine.enrich(policy_query, full_text)
                if enriched.enriched:
                    full_text = enriched.enriched
            except Exception:
                logger.info("v22 decision engine enrichment failed", exc_info=True)

        # ── v22: Semantic Cache — store response ─────────────────────
        if self._semantic_cache is not None:
            try:
                self._semantic_cache.put(policy_query, full_text, source="local")
            except Exception:
                logger.info("v22 semantic cache put failed", exc_info=True)

        # ── v22: Preference learning ─────────────────────────────────
        if self._preference_store is not None:
            try:
                self._preference_store.learn_from_query_pattern(policy_query)
            except Exception:
                logger.info("v22 preference learning failed", exc_info=True)

        final_text, saved_report_path = self._maybe_export_report(policy_query, full_text)
        if saved_report_path:
            logger.info("Saved long report: %s", saved_report_path)

        # Skip the end-of-turn vet pass when the stream vetter already
        # delivered a clarifier — a second rewrite would produce a new
        # clarifier ("I'm not sure I heard you right…" etc.) that conflicts
        # with the one the user already heard streaming.
        if self._response_vetter is not None and not self._turn_vetter_rewrote:
            try:
                confidence = 0.6
                if self._confidence_engine is not None:
                    try:
                        confidence = float(
                            self._confidence_engine.score(policy_query, final_text)
                            or 0.6
                        )
                    except Exception:
                        confidence = 0.6
                vetted = self._response_vetter(policy_query, final_text, confidence)
                if isinstance(vetted, str) and vetted.strip() and vetted != final_text:
                    logger.info(
                        "Router guardrail rewrote LLM reply (conf=%.2f): '%s' -> '%s'",
                        confidence, final_text[:80], vetted[:80],
                    )
                    final_text = vetted
            except Exception:
                logger.debug("response_vetter failed", exc_info=True)

        # Sprint J: Jarvis Offer Protocol -- synthesise an actionable
        # follow-up offer ("Want me to do that for you, Boss?") from
        # the user's original query, stash the matching action in the
        # OfferRegistry so a single "yes" on the next turn fires it
        # without an LLM round-trip, and append the offer line to the
        # spoken reply. We deliberately keep the *cached* response
        # bare (offer-free) so subsequent identical queries don't
        # accumulate compounding offers from the cache.
        spoken_text = final_text
        try:
            from core.cognitive.offer_synthesizer import synthesize_offer
            from core.router.offer_registry import get_offer_registry

            proposal = synthesize_offer(text, final_text)
            if proposal is not None:
                spoken_text = self._append_offer_to_reply(
                    final_text, proposal.offer_text,
                )
                get_offer_registry().stash(
                    action=proposal.action,
                    args=proposal.args,
                    offer_text=proposal.offer_text,
                    source_query=text,
                    source_response=final_text,
                    metadata={
                        "category": proposal.category,
                        "source": "local_llm",
                    },
                )
        except Exception:
            logger.debug("Jarvis offer synth failed", exc_info=True)
            spoken_text = final_text

        if should_buffer_response:
            self._bus.emit_long("response_ready", text=spoken_text)
        elif react_step > 0:
            self._bus.emit_long(
                "partial_response",
                text=spoken_text,
                is_first=True,
                is_last=True,
                source="local",
            )

        self._emit_final_metrics(t0_total, first_token_ms, final_text, trace_id=trace_id)

        self._bus.emit(
            "cursor_response",
            query=text.lower().strip(),
            response=final_text,
        )

        self._bus.emit("llm_response_complete", text=final_text)

        try:
            from core.cognition.predictor import predict_next_queries
            from core.rag.prefetch_engine import (
                RagPrefetchEngine,
                merge_prefetch_candidates,
                predict_followup_queries,
            )
            if self._rag_engine is not None and self._current_runtime_mode != "SECURE":
                v7 = self._config.get("v7_intelligence") or {}
                if (
                    bool(v7.get("prediction_prefetch_enabled", True))
                    and self._prediction_prefetch_enabled()
                ):
                    tsnips: list[str] = []
                    active_task: dict[str, Any] | str | None = None
                    if self._timeline is not None:
                        try:
                            tsnips = self._timeline.context_snippets_for_prediction()
                            active_task = self._timeline.get_last_active_task()
                        except Exception:
                            logger.debug('Timeline prediction snippets read failed', exc_info=True)
                    last_proj = None
                    recent_ent: list[dict[str, Any]] = []
                    if self._memory_graph is not None:
                        try:
                            last_proj = self._memory_graph.get_last_active_project()
                            recent_ent = self._memory_graph.get_recent_entities(8)
                        except Exception:
                            logger.debug('Timeline prediction snippets read failed', exc_info=True)
                    pred_ctx = {
                        "last_queries": list(self._recent_queries),
                        "active_task": active_task,
                        "recent_actions": tsnips,
                        "timeline_snippets": tsnips,
                        "last_project": last_proj,
                        "recent_entities": recent_ent,
                        "feedback_engine": self._feedback_engine,
                    }
                    predicted = predict_next_queries(pred_ctx)
                    legacy = predict_followup_queries(text, history or [])
                    pf_cfg = (v7.get("prefetch") or {})
                    max_pf = int(pf_cfg.get("max_prefetch_candidates", 12))
                    merged = merge_prefetch_candidates(
                        predicted, legacy, max_candidates=max_pf,
                    )
                    if self._timeline is not None:
                        try:
                            hint = self._timeline.suggest_next_from_pattern()
                            if hint:
                                merged = merge_prefetch_candidates(
                                    [hint], merged, max_candidates=max_pf,
                                )
                        except Exception:
                            logger.debug('Timeline prediction snippets read failed', exc_info=True)
                    self._prev_predictions = list(merged[:12])
                    if self._feedback_engine is not None:
                        try:
                            self._feedback_engine.record_prefetch_scheduled(len(merged))
                        except Exception:
                            logger.debug('Timeline prediction snippets read failed', exc_info=True)
                    eng = self._prefetch_engine or RagPrefetchEngine(
                        self._rag_engine, self._config,
                    )
                    eng.schedule_fire_and_forget(
                        merged,
                        gpu_util_pct=gpu_util,
                        prediction_accuracy=pred_acc,
                    )
        except Exception:
            logger.debug('Memory graph project lookup failed', exc_info=True)

        try:
            if self._suggester is not None and self._timeline is not None:
                acc = 0.5
                if self._feedback_engine is not None:
                    acc = float(
                        self._feedback_engine.compute_accuracy_metrics().get(
                            "prediction_accuracy", 0.5,
                        ),
                    )
                for sug in self._suggester.produce(
                    self._timeline,
                    prediction_accuracy=acc,
                    last_query=text,
                ):
                    self._bus.emit_fast("v7_suggestion", text=sug)
        except Exception:
            logger.debug('Timeline pattern hint merge failed', exc_info=True)

    def _vet_stream_sentence(self, policy_query: str | None, text: str) -> str:
        """Run the router guardrail on a single streaming sentence.

        The streaming path emits ``partial_response`` events token-by-token,
        so bad LLM output reaches TTS long before the end-of-stream vetter
        at ``response_ready`` can see it. This hook applies the same
        ``vet_llm_response`` pass per-sentence, catching action-promise
        hallucinations inside quoted wrappers like ``The answer is "..."``
        before we speak them.

        Returns the (possibly rewritten) sentence. Falls back to ``text``
        unchanged if no vetter is attached or vetting raises.

        Side effect: flips ``self._turn_vetter_rewrote`` to True on any real
        rewrite so end-of-turn logic can break the cascade (skip strict
        recovery when we already delivered a safe clarifier).
        """
        if self._response_vetter is None or not text:
            return text
        query = policy_query or ""
        if not query:
            return text
        try:
            vetted = self._response_vetter(query, text, 0.6)
            if isinstance(vetted, str) and vetted.strip() and vetted != text:
                logger.info(
                    "Streaming guardrail rewrote sentence: '%s' -> '%s'",
                    text[:80], vetted[:80],
                )
                self._turn_vetter_rewrote = True
                return vetted
        except Exception:
            logger.debug("Streaming vetter failed", exc_info=True)
        return text

    async def _run_llm_streaming(
        self,
        prompt: str,
        t0_total: float,
        *,
        emit_partial: bool = True,
        model_role: str | None = None,
        max_tokens_override: int | None = None,
        policy_query: str | None = None,
        extra_stop_sequences_override: tuple[str, ...] | None = None,
        _watchdog_guard: bool = False,
    ) -> tuple[str, float, bool]:
        """Run LLM inference with optional streaming to TTS.

        Returns (full_response_text, first_token_ms, was_preempted).
        When emit_partial=True, sentences are streamed to TTS in real-time.
        When False (ReAct follow-up), we collect silently.
        """
        if self._runtime_watchdog is not None and not _watchdog_guard:
            watched = await self._runtime_watchdog.run_async(
                "llm_inference",
                self._run_llm_streaming(
                    prompt,
                    t0_total,
                    emit_partial=emit_partial,
                    model_role=model_role,
                    max_tokens_override=max_tokens_override,
                    policy_query=policy_query,
                    extra_stop_sequences_override=extra_stop_sequences_override,
                    _watchdog_guard=True,
                ),
                default=("", 0.0, True),
                metadata={"prompt_chars": len(prompt)},
            )
            return watched.value

        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[tuple[str, bool]] = asyncio.Queue()
        first_token_time: list[float | None] = [None]

        def _on_token(token_text: str, is_done: bool) -> None:
            if first_token_time[0] is None and token_text:
                first_token_time[0] = time.perf_counter()
            # Thread-safe push to asyncio queue (eliminates polling latency)
            loop.call_soon_threadsafe(token_queue.put_nowait, (token_text, is_done))

        generate_kwargs: dict[str, Any] = {"on_token": _on_token}
        if model_role:
            generate_kwargs["model_role"] = model_role
        if max_tokens_override:
            generate_kwargs["max_tokens_override"] = int(max_tokens_override)
        # Sprint C4/K: token-layer stop sequences for FAST. K removed
        # "(" because it killed valid replies on token 1; the streaming
        # sanitiser strips parentheticals safely after generation.
        if extra_stop_sequences_override is not None:
            generate_kwargs["extra_stop_sequences"] = extra_stop_sequences_override
        elif str(model_role or "").lower() == "fast":
            try:
                from brain.mlx_llm import _FAST_PATH_STOP_SEQUENCES
                generate_kwargs["extra_stop_sequences"] = (
                    _FAST_PATH_STOP_SEQUENCES
                )
            except Exception:
                logger.debug(
                    "FAST stop-sequences import failed", exc_info=True,
                )
        generate_task = asyncio.create_task(
            self._llm.generate_streaming(prompt, **generate_kwargs)
        )

        # Thinking earcon: schedule a soft click at 1.2s if no first token
        # AND no TTS emission yet. Cancelled when we actually emit content.
        # Only when emit_partial is True (i.e. user-facing turn).
        thinking_earcon_task: asyncio.Task | None = None
        if emit_partial:
            async def _thinking_earcon_delay() -> None:
                try:
                    await asyncio.sleep(1.2)
                    if first_token_time[0] is None:
                        try:
                            self._bus.emit("thinking_earcon")
                        except Exception:
                            pass
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.debug("Thinking earcon scheduler failed", exc_info=True)
            thinking_earcon_task = asyncio.create_task(_thinking_earcon_delay())

        stream_id = uuid.uuid4().hex
        sentence_buffer = ""
        sentences_emitted = 0
        full_response_parts: list[str] = []
        trailing_sentence = ""
        try:
            while True:
                # Wait for the next token without polling
                get_task = asyncio.create_task(token_queue.get())
                done, pending = await asyncio.wait(
                    [get_task, generate_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                if get_task in done:
                    token_text, is_done = get_task.result()
                else:
                    # generate_task finished but queue might not be empty
                    get_task.cancel()
                    try:
                        token_text, is_done = token_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                if is_done:
                    trailing_sentence = sentence_buffer.strip()
                    break

                if not token_text:
                    continue

                sentence_buffer += token_text

                if emit_partial:
                    # First-chunk acceleration: for the VERY first emission
                    # we allow a clause boundary (comma / semicolon / em-dash)
                    # so TTS starts speaking ~1s sooner on longer first
                    # sentences. Subsequent chunks stick to full sentences.
                    ready = None
                    if sentences_emitted == 0:
                        ready = self._extract_first_clause(sentence_buffer)
                    if ready is None:
                        ready = self._extract_complete_sentence(sentence_buffer)
                    if ready:
                        sentence_text, remainder = ready
                        safe_sentence = self._sanitize_emittable_text(sentence_text)
                        if safe_sentence:
                            vet_fn = getattr(self, "_vet_stream_sentence", None)
                            if callable(vet_fn):
                                safe_sentence = vet_fn(policy_query, safe_sentence)
                        if safe_sentence:
                            sentences_emitted += 1
                            self._bus.emit_long(
                                "partial_response",
                                text=safe_sentence,
                                is_first=(sentences_emitted == 1),
                                is_last=False,
                                source="local",
                                stream_id=stream_id,
                            )
                            full_response_parts.append(safe_sentence)
                            try:
                                self._turn_emitted_sentences.append(safe_sentence)
                            except Exception:
                                pass
                        sentence_buffer = remainder

            while not token_queue.empty():
                try:
                    tok, done_flag = token_queue.get_nowait()
                    if tok:
                        sentence_buffer += tok
                    if done_flag:
                        trailing_sentence = sentence_buffer.strip()
                        break
                except asyncio.QueueEmpty:
                    break

            result = await generate_task
            answer, preempted = result

            if not trailing_sentence:
                trailing_sentence = sentence_buffer.strip()

            if trailing_sentence and not preempted:
                safe_trailing = self._sanitize_emittable_text(trailing_sentence)
                if safe_trailing:
                    vet_fn = getattr(self, "_vet_stream_sentence", None)
                    if callable(vet_fn):
                        safe_trailing = vet_fn(policy_query, safe_trailing)
                if safe_trailing:
                    sentences_emitted += 1
                    if emit_partial:
                        self._bus.emit_long(
                            "partial_response",
                            text=safe_trailing,
                            is_first=(sentences_emitted == 1),
                            is_last=True,
                            source="local",
                            stream_id=stream_id,
                        )
                    full_response_parts.append(safe_trailing)
                    try:
                        self._turn_emitted_sentences.append(safe_trailing)
                    except Exception:
                        pass

            raw_full_text = " ".join(full_response_parts).strip() if full_response_parts else answer
            full_text = " ".join(full_response_parts).strip() if full_response_parts else self._sanitize_emittable_text(answer)
            if not full_text and raw_full_text:
                logger.warning(
                    "LLM output sanitized to empty: %s",
                    self._compact_text(raw_full_text)[:240],
                )
                self._bus.emit("metrics_event", counter="llm_sanitized_empty")

            if sentences_emitted == 0 and full_text and emit_partial and not preempted:
                vet_fn = getattr(self, "_vet_stream_sentence", None)
                vetted_full = vet_fn(policy_query, full_text) if callable(vet_fn) else full_text
                if vetted_full:
                    if vetted_full != full_text:
                        full_text = vetted_full
                    self._bus.emit_long(
                        "partial_response",
                        text=full_text,
                        is_first=True,
                        is_last=True,
                        source="local",
                        stream_id=stream_id,
                    )

            first_token_ms = (
                (first_token_time[0] - t0_total) * 1000
                if first_token_time[0] is not None
                else 0.0
            )

            return full_text, first_token_ms, preempted
        except asyncio.CancelledError:
            generate_task.cancel()
            try:
                await generate_task
            except Exception:
                logger.debug('Metrics counter emit failed', exc_info=True)
            raise
        finally:
            if thinking_earcon_task is not None and not thinking_earcon_task.done():
                thinking_earcon_task.cancel()

    def _emit_final_metrics(
        self,
        t0: float,
        first_token_ms: float,
        full_text: str,
        trace_id: str | None = None,
    ) -> None:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_calls += 1
        word_count = len(full_text.split()) if full_text else 0
        self._total_tokens_approx += word_count

        if first_token_ms > 0:
            self._first_token_latencies.append(first_token_ms)

        self._bus.emit("metrics_latency", name="llm", ms=elapsed_ms)
        if first_token_ms > 0:
            self._bus.emit("metrics_latency", name="llm_first_token", ms=first_token_ms)

        try:
            from core.observability.per_module_latency import get_latency_board

            get_latency_board().record_module_call(
                getattr(self, "_latency_board_llm", "llm_large"),
                float(elapsed_ms),
                error=False,
            )
        except Exception:
            logger.debug('Generate task await after cancel failed', exc_info=True)

        try:
            from core.unified_trace import new_trace
            ut = new_trace(trace_id)
            ut.latency_ms["llm_total"] = elapsed_ms
            if first_token_ms > 0:
                ut.latency_ms["llm_first_token"] = first_token_ms
            ut.decision_path.extend(["llm_stream", "react_loop"])
            self._bus.emit("v7_unified_trace", **ut.to_dict())
        except Exception:
            logger.debug('Generate task await after cancel failed', exc_info=True)

        logger.info(
            "Brain: %.0fms total, %.0fms first-token, %d words, %d tool calls this turn",
            elapsed_ms, first_token_ms, word_count, self._total_tool_calls,
        )

        if self._feedback_engine is not None and self._total_calls % 25 == 0:
            try:
                fm = self._feedback_engine.compute_accuracy_metrics()
                logger.info("v7_feedback_metrics %s", fm)
            except Exception:
                logger.debug('LLM latency board record failed', exc_info=True)

    @staticmethod
    def _extract_complete_sentence(buffer: str) -> tuple[str, str] | None:
        """Extract the first complete sentence from the buffer.

        A sentence ends with . ! or ? followed by a space, or
        a buffer 60+ chars ending with punctuation.
        """
        match = _SENTENCE_BOUNDARY.search(buffer)
        if match:
            split_pos = match.end()
            return buffer[:split_pos].rstrip(), buffer[split_pos:]

        if len(buffer) >= 60 and _SENTENCE_END.search(buffer.rstrip()):
            return buffer.rstrip(), ""

        return None

    @staticmethod
    def _extract_first_clause(buffer: str) -> tuple[str, str] | None:
        """Fast-path for the FIRST chunk: emit on a clause break to cut
        perceived first-audio latency from ~4s to ~1.5s.

        Matches a comma / semicolon / em-dash followed by whitespace, but
        only when the buffer is already >= 24 characters (so we don't
        truncate a short prefix like "Newton," into its own slice).

        Used exclusively when ``sentences_emitted == 0``. Subsequent chunks
        fall back to full-sentence boundaries for better prosody.
        """
        if len(buffer) < 24:
            return None
        # Skip the first 20 chars so we never cut a greeting like "Hi, Boss"
        # in half — we want a natural mid-sentence break.
        match = re.search(r"[,;—](?:\s|$)", buffer[20:])
        if not match:
            return None
        split_pos = 20 + match.end()
        head = buffer[:split_pos].rstrip()
        tail = buffer[split_pos:]
        # Don't split on a micro-fragment like "Okay," or "Well,"
        head_words = head.split()
        if len(head_words) < 4:
            return None
        return head, tail

    def close(self) -> None:
        avg_first_token = (
            sum(self._first_token_latencies) / len(self._first_token_latencies)
            if self._first_token_latencies else 0
        )
        logger.info(
            "Brain stats: %d calls, ~%d tokens, %d tool calls, "
            "%d react loops, avg first-token %.0fms",
            self._total_calls, self._total_tokens_approx,
            self._total_tool_calls, self._total_react_loops,
            avg_first_token,
        )
        close_fn = getattr(self._llm, "close", None)
        if callable(close_fn):
            close_fn()
        else:
            self._llm.shutdown()

    def get_stats(self) -> dict:
        avg_first_token = (
            sum(self._first_token_latencies) / len(self._first_token_latencies)
            if self._first_token_latencies else 0
        )
        return {
            "available": self.available,
            "loaded": self.is_loaded,
            "total_calls": self._total_calls,
            "total_tokens_approx": self._total_tokens_approx,
            "total_tool_calls": self._total_tool_calls,
            "total_react_loops": self._total_react_loops,
            "avg_first_token_ms": round(avg_first_token, 1),
        }
