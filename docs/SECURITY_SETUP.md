# ATOM — Security setup (credentials, tiers, audit)

**Platform:** macOS. This complements `config/settings.json` and `core/security_policy.py`.

## Credentials (single setup path)

Use the interactive helper — it stores secrets via `CredentialManager` (encrypted), not plain text in the repo:

```bash
python3 scripts/setup_api_keys.py
```

- **Master password:** set `ATOM_MASTER_PASSWORD` for non-interactive runs, or enter when prompted.
- Optional env vars for automated setup are documented in the script header (`GEMINI_FAST_API_KEY`, etc.).

Do not commit API keys or the master password.

## Permission tiers vs `security.mode`

Intent actions are classified into **tiers 1–4** (`core/security_tiers.py`). **`security.mode`** sets the maximum tier allowed before other gates (features, lock mode, confirmation):

| `security.mode` | Max tier | Effect |
|-----------------|----------|--------|
| `strict` (default) | 3 | Tier-4 actions (e.g. shutdown, kill_process, empty_recycle_bin) are **denied** at policy gate. |
| `standard`, `balanced`, `permissive`, `development` | 4 | Tier-4 actions may proceed if they pass feature flags, lock mode, confirmation, and behavior checks. |

Use **`strict`** on shared or untrusted machines. Use **`standard`** only when you accept power and high-impact actions behind confirmations.

## Other knobs

- **`control.lock_mode`:** `open` / `restricted` / `secure` / `paranoid` — session and device binding (see `core/lock_modes.py`).
- **`features`:** Disables whole classes (e.g. `desktop_control`, `file_ops`).
- **`security.require_confirmation_for`:** Extra voice/UI confirmation for listed intents.

## Audit logs

| Log | Contents |
|-----|----------|
| `logs/audit.log` | Security policy allow/deny (`SecurityPolicy.audit_log`). |
| `logs/autonomy.log` | Autonomous habit / rule decisions. |
| `logs/proactive_insights.log` | Proactive `jarvis_insight` quota audit (if enabled). |

Ensure `logs/` is not world-readable in multi-user environments (`audit.log` is chmod `0600` when supported).
