"""
ATOM -- Reasoning Engine (Agentic Tool Use + Planning + Code Execution).

The intelligence layer that transforms ATOM from a regex remote control
into a JARVIS-level reasoning system.

Components:
    ToolRegistry      -- Formal definitions of all 40+ tools ATOM can use
    ToolParser        -- Extracts tool calls from LLM output (multi-format)
    ActionExecutor    -- Security-gated bridge from tool calls to system actions
    ReasoningPlanner  -- Multi-step task decomposition and execution
    CodeSandbox       -- Safe Python expression evaluation
"""
