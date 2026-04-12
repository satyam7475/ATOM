"""
ATOM OS -- Vision Perception Layer
Allows ATOM to 'see' the user's screen using native macOS APIs and Cloud Vision ML models.
"""

from __future__ import annotations

import os
import time
import subprocess
import logging
from typing import Optional

logger = logging.getLogger("atom.perception.vision")

class ScreenReader:
    """Mac-native screencapture integrated with Vision AI."""
    
    def __init__(self, gemini_client=None):
        self.gemini = gemini_client
        self.temp_dir = "/tmp/atom_vision"
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def capture_screen(self) -> str:
        """Takes a physical macOS screenshot and returns the file path."""
        ts = int(time.time())
        filepath = f"{self.temp_dir}/screen_{ts}.png"
        try:
            # -x: no sound, -C: capture cursor
            subprocess.run(["screencapture", "-x", "-C", filepath], check=True)
            logger.info(f"Screen physically captured -> {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return ""
            
    async def analyze_screen(self, query: str = "Analyze the UI layout and read the text on the screen.") -> str:
        """Pipes physical screenshot to Cloud Vision (Gemini) for deep layout comprehension."""
        filepath = self.capture_screen()
        if not filepath or not os.path.exists(filepath):
            return "Vision subsystem failed: Could not capture physical display."
            
        if not self.gemini:
            return "Vision subsystem fallback: Screen captured locally, but Gemini Client offline."
            
        try:
            # Requires the Gemini Client to accept image paths in its .ask() or .analyze()
            result = await self.gemini.ask(query, image_path=filepath)
            
            # Clean up cache so /tmp doesn't blow up
            os.remove(filepath)
            
            return result
        except Exception as e:
            logger.error(f"Vision Inference Failed: {e}")
            return f"Vision Error: {str(e)}"
