"""ATOM -- Proactive skills package.

Small, single-purpose services that emit spoken content on their own
schedule (not tied to a user turn). Each service is opt-in via
``config["morning_briefing"]`` / ``config["whats_on_my_plate"]`` /
``config["routine_triggers"]`` and writes state to ``data/``.
"""
