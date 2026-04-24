"""Cross-device integrations (Phase 1: iPhone Shortcuts bridge).

The bridge runs a small local HTTP listener so an iPhone -- using the
built-in Shortcuts app, no Swift, no Xcode -- can POST Face ID
verifications, presence changes, and named triggers to ATOM.

Public surface:

``iphone_bridge.IPhoneBridge``
    aiohttp listener with token auth, port fallback, and single-device
    lock. Starts lazily; if the port is taken or aiohttp is missing it
    fails-soft and the rest of ATOM keeps booting.

``bridge_auth``
    Token generation (first-boot), constant-time validation, and the
    append-only auth-failure audit log.

``trusted_device``
    Single trusted iPhone UDID-hash. First handshake wins; subsequent
    devices are rejected with a 409 until an explicit reset.

The HTTP contract is intentionally trivial so a future SwiftUI companion
app can reuse the endpoints verbatim (Phase 1.5 / 3).
"""

from __future__ import annotations
