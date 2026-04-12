"""
ATOM - Desktop Agent (Recursive Macro Execution Loop)

Orchestrates multi-step computer automation by fusing the Neural Engine ScreenReader,
macOS Accessibility state, and Gemini reasoning to iteratively drive the UI 
until the goal is achieved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger("atom.desktop_agent")

class DesktopAgent:
    def __init__(self, gemini_client: Any, router: Any, config: dict):
        self._gemini = gemini_client
        self._router = router
        self._config = config
        self._max_iterations = 15

    async def execute_macro(self, goal: str) -> str:
        """Run a recursive observation-action loop until the goal is complete."""
        if not self._gemini or not self._gemini.is_available:
            return "Cannot execute macro: Cloud intelligence is required for continuous reasoning."

        from context.screen_reader import ScreenReader
        from core.desktop_control import describe_focused_element

        screen_reader = ScreenReader(self._config)
        
        logger.info("Starting Desktop Agent Macro: %s", goal)
        
        # We need the ToolRegistry schema to give to Gemini
        from core.reasoning.tool_registry import get_tool_registry
        
        # Build the system instruction for the agent
        system_instruction = (
            "You are ATOM, an autonomous agent driving a macOS computer to achieve the user's goal.\n"
            f"GOAL: {goal}\n\n"
            "INSTRUCTIONS:\n"
            "1. You will be provided with the CURRENT SCREEN STATE (OCR Text + Focused Element) at every step.\n"
            "2. You must decide the next logical physical action (e.g., clicking, typing, hotkeys) to progress the goal.\n"
            "3. Output ONLY a valid JSON tool call matching the ToolRegistry schema below.\n"
            "4. If the goal is complete, yield the 'macro_complete' action.\n\n"
            "TOOL SCHEMA:\n"
            "You can ONLY use the tools listed here. Return JSON like: {\"action\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
            "Available Actions:\n"
            " - click_ui_element (arguments: label)\n"
            " - set_focused_text (arguments: text)\n"
            " - hotkey_combo (arguments: combo)\n"
            " - press_key (arguments: key)\n"
            " - open_app (arguments: name)\n"
            " - type_text (arguments: text)\n"
            " - scroll_down / scroll_up\n"
            " - macro_complete (Use this when the goal is done)\n"
        )
        
        history_log = []
        
        for iteration in range(self._max_iterations):
            # 1. OBSERVE
            screen_data = screen_reader.capture_and_read()
            focused_ui = describe_focused_element()
            
            # Trim OCR text aggressively to save tokens and context window
            ocr_text = screen_data.get("text", "")
            if len(ocr_text) > 2000:
                ocr_text = ocr_text[:2000] + "..."
                
            state_prompt = (
                f"--- ITERATION {iteration + 1} ---\n"
                f"FOCUSED UI: {focused_ui}\n"
                f"OCR SCREEN TEXT: {ocr_text}\n"
                f"PREVIOUS ACTIONS: {', '.join(history_log[-3:]) if history_log else 'None'}\n"
                "What is your next JSON tool action?"
            )
            
            logger.info("Macro iteration %d: Asking Gemini...", iteration + 1)
            
            # 2. THINK (Ask Gemini)
            # Increase timeout for complex reasoning
            try:
                # We'll pass the system instruction as part of the prompt if the client doesn't support it directly
                full_prompt = f"{system_instruction}\n\n{state_prompt}"
                
                # Gemini free tier uses very aggressive rate limits, sleep briefly
                if iteration > 0:
                    await asyncio.sleep(2.0)
                    
                response_text, ok = await self._gemini.ask(full_prompt, max_tokens=200)
                if not ok or not response_text:
                    return f"Macro stopped at step {iteration + 1}: Brain disconnected."
                
                # 3. PARSE
                response_clean = response_text.strip("```json \n").strip()
                try:
                    action_data = json.loads(response_clean)
                    action_name = action_data.get("action", "")
                    arguments = action_data.get("arguments", {})
                except Exception:
                    logger.warning("Agent returned invalid JSON: %s", response_clean)
                    history_log.append("Error: Invalid JSON returned")
                    continue
                
                logger.info("Agent chose action: %s %s", action_name, arguments)
                
                # 4. ACT
                if action_name == "macro_complete":
                    return f"Macro complete after {iteration + 1} steps."
                
                if not action_name:
                    history_log.append("Error: Empty action")
                    continue
                    
                # SECURITY GATE: Enforce autonomous safety bounds
                from core.security.action_signing import merge_signed_args
                sargs = merge_signed_args(self._security, action_name, arguments)
                allowed, reason = self._security.allow_action(action_name, sargs)
                if not allowed:
                    logger.warning("Desktop Agent attempted blocked action %s: %s", action_name, reason)
                    history_log.append(f"Error Action Blocked: {reason}")
                    continue
                
                # We use the router's action dispatch to safely execute tools
                dispatch = getattr(self._router, "_ACTION_DISPATCH", {})
                if action_name in dispatch:
                    handler = dispatch[action_name]
                    result = handler(self._router, action_name, arguments)
                    history_log.append(f"{action_name} -> {result}")
                else:
                    logger.warning("Unrecognized action requested by agent: %s", action_name)
                    history_log.append(f"Error: Unknown action {action_name}")
            except Exception as e:
                logger.error("Macro loop crash: %s", e, exc_info=True)
                return f"Macro crashed: {e}"
                
        return f"Macro timed out after {self._max_iterations} limits. Goal not fully reached."
