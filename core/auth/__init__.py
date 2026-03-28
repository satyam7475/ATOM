"""
ATOM -- Owner Authentication System.

Biometric + behavioral authentication for owner exclusivity.
JARVIS only responds to Tony. ATOM only responds to Satyam.
"""

from core.auth.voice_auth import VoicePrintAuth
from core.auth.behavior_auth import BehavioralAuth

__all__ = ["VoicePrintAuth", "BehavioralAuth"]
