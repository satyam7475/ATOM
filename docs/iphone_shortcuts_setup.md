# iPhone Shortcuts -> ATOM bridge (Phase 1 setup)

This doc walks you through turning your iPhone into a trusted second
device for ATOM using only the built-in Shortcuts app. No Xcode, no
Apple developer account, no third-party apps.

When you are done:

- Your iPhone can Face-ID verify you, and ATOM will treat tier-3
  (destructive) voice commands as authorised for 5 minutes.
- Tapping "I'm at desk" on your iPhone lets ATOM greet you the moment
  you sit down.
- Running "Morning routine" on iPhone triggers ATOM's briefing.

All traffic stays on your local network. Nothing goes to Apple or
any cloud. The shared secret that protects the bridge lives in iCloud
Keychain, which means you paste it into the Shortcut **once** and it
syncs to every Apple device you own.

---

## 0. Before you start

On your **Mac**:

1. Make sure `config/settings.json` has the bridge enabled:

   ```json
   "cross_device": {
       "enabled": true,
       "bridge_port": 8787,
       "bind_host": "127.0.0.1"
   }
   ```

   The default `bind_host` is `127.0.0.1` (localhost). If you want
   your iPhone to hit ATOM over Wi-Fi from a different device, change
   this to your Mac's LAN IP (e.g. `192.168.1.42`). Localhost-only
   is the safer default while you are getting started -- you can test
   on the Mac with `curl` first.

2. Start ATOM once. The bridge mints a token at `config/bridge_token`
   on first boot and writes the actual bound port to
   `logs/atom_bridge.port`:

   ```bash
   cat config/bridge_token
   # xxxxxxxxxxxxxxxxxxxxxxxxxxxx (43 urlsafe chars)

   cat logs/atom_bridge.port
   # 8787   (or 8788 / 8789 if 8787 was busy)
   ```

3. Health-check the bridge from the Mac:

   ```bash
   curl -s http://127.0.0.1:$(cat logs/atom_bridge.port)/health
   # {"ok": true, "version": 1}
   ```

If that returns `{"ok": true}`, the Mac side is ready.

Find the Mac's LAN IP for your phone to hit:

```bash
ipconfig getifaddr en0   # Wi-Fi
# or
ipconfig getifaddr en1   # Ethernet / wired dock
```

Write that IP down. You will paste it into the Shortcuts below.

---

## 1. Store the bridge token in iCloud Keychain

Apple's Keychain is the only shared-secret store that's available to
the stock Shortcuts app. Putting the token there means:

- It's encrypted at rest.
- It syncs to every Apple device you sign into -- so swapping iPhones
  is just signing into iCloud, no re-setup.
- Shortcuts can read it via the `Get Password for Account...` action.

Steps, on your iPhone:

1. Open **Settings -> Passwords** (iOS 17+: **Settings -> Passwords**;
   older iOS: **Settings -> Passwords & Accounts -> Website &
   App Passwords**).
2. Tap **+** (add) at the top right.
3. Fill in:
   - **Website / App Name:** `atom-bridge`
   - **User Name:** `atom`
   - **Password:** paste the value of `config/bridge_token` from your
     Mac
4. Save.

Done. In the Shortcut actions below you will use
**Get Details of Saved Passwords** with account `atom` to retrieve
this.

---

## 2. Shortcut 1 -- "Verify with ATOM" (Face ID freshness)

This is the most important Shortcut. Running it prompts Face ID and,
on success, POSTs to `/faceid` so ATOM's router unlocks tier-3
actions for 5 minutes.

### Actions (add in order)

1. **Get Device Details** -> *Name*. (We'll use the device name as
   the stable id the bridge hashes.)
2. **Set Variable** -> name it `device_id`, value = the output of
   step 1.
3. **Get Current Date**. Save as `now`.
4. **Authenticate with Face ID / Touch ID**
   - Prompt text: `Verify with ATOM`
   - On success: continue.
   - On cancel: Stop and notify the user.
5. **Get Password for atom-bridge** (search "Passwords" in the
   action picker). Returns the token.
6. **Get Contents of URL**
   - URL: `http://<MAC_LAN_IP>:<PORT>/faceid`
   - Method: **POST**
   - Request Body: **JSON**
   - Headers:
     - `X-ATOM-Token` -> `Magic Variable` = the token from step 5
     - `Content-Type` -> `application/json`
   - Body:
     ```json
     {
       "device_id": "[device_id]",
       "verified": true,
       "label": "[device_name]",
       "timestamp": "[now]"
     }
     ```
     (Tap each bracketed value and replace with the Magic Variable
     from the earlier steps.)
7. **Show Result** -> `Contents of URL`. (Optional; lets you sanity
   check `{"ok": true}`. You can delete this after a successful
   first run.)

Save the Shortcut as **Verify with ATOM** and add it to your Home
Screen + Lock Screen + Control Center for one-tap access.

---

## 3. Shortcut 2 -- "I'm at desk" (presence change)

Same shape, minus the Face ID step. No verification -- presence is
not a security signal, it is a proactivity signal.

Actions:

1. Get Device Details -> Name -> Set variable `device_id`.
2. Get Password for `atom-bridge`.
3. Get Contents of URL:
   - URL: `http://<MAC_LAN_IP>:<PORT>/presence`
   - Method: POST
   - JSON body:
     ```json
     {
       "device_id": "[device_id]",
       "state": "at_desk"
     }
     ```
   - Header: `X-ATOM-Token` = token from step 2.

Save as **I'm at desk**. Add a sibling Shortcut named **I'm leaving**
with `"state": "leaving"` for the opposite direction.

Allowed values for `state`: `at_desk`, `leaving`, `home`, `away`,
`busy`. The bridge rejects anything else with a 400.

---

## 4. Shortcut 3 -- "Run morning routine" (named trigger)

Named triggers let iPhone fire any ATOM background routine by name.

Actions:

1. Get Device Details -> Name -> Set variable `device_id`.
2. Get Password for `atom-bridge`.
3. Get Contents of URL:
   - URL: `http://<MAC_LAN_IP>:<PORT>/trigger`
   - Method: POST
   - JSON body:
     ```json
     {
       "device_id": "[device_id]",
       "name": "morning_routine"
     }
     ```
   - Header: `X-ATOM-Token` = token from step 2.

Save as **Run morning routine**. Use an **Automation** (Shortcuts app
-> Automation tab) to fire this at 07:00 on weekdays -- fully hands-off.

Other trigger names ATOM will understand in Pillar 2 (Phase 1): `evening_wrap`,
`focus_on`, `focus_off`. Unknown names are accepted with 200 but
silently ignored, so you can start using a name before wiring it up.

---

## 5. Verify the whole loop

On the iPhone, run **Verify with ATOM**. Face ID prompts; after you
look at the screen, the Shortcut should show `{"ok": true, "accepted_at": ...}`.

On the Mac, tail the audit log:

```bash
tail -f logs/atom_bridge_audit.jsonl
```

You should see **no lines** (audit log only fires on auth failures
and 409s). If every Shortcut run writes an entry here with
`reason="bad_token"`, the token in Keychain is wrong. Copy
`config/bridge_token` again and replace the Keychain entry exactly
-- note that `token_urlsafe` uses `-` and `_` which some copy-paste
tools mangle.

---

## 6. Reset the trusted device

If you want to point the bridge at a different iPhone:

```bash
rm data/trusted_iphone.json
```

The next successful POST will claim the slot for that phone. The
bridge itself does not need a restart.

---

## 7. Security notes

- **HTTP, not HTTPS.** Phase 1 ships plain HTTP. The pre-shared
  token is enough on your home network, but **do not expose the
  bridge port on public Wi-Fi**. If you have to work off a cafe
  network, bind `127.0.0.1` only and use `I'm at desk` Shortcuts
  you keep on airplane mode until you are home. Phase 1.5 adds
  self-signed TLS.
- **Token rotation.** Delete `config/bridge_token` on the Mac; the
  next boot mints a new one. Then re-paste into iCloud Keychain on
  the iPhone.
- **Rate limit.** The bridge accepts one request per second per
  endpoint per device. Running the same Shortcut twice in a second
  gets a 429; the Shortcut will still complete but ATOM ignores the
  duplicate. No state corruption.
- **Flood detection.** 10 bad-token attempts in any rolling
  60-second window triggers a spoken warning ("Something's probing
  the bridge") so a brute-forcing attacker can't do it quietly.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Could not connect to host` | Mac asleep; wrong IP; wrong port | Wake Mac, check `ipconfig getifaddr en0`, check `cat logs/atom_bridge.port` |
| `{"ok": false, "error": "unauthorized"}` | Token mismatch | Re-paste `config/bridge_token` into iCloud Keychain |
| `{"ok": false, "error": "device_conflict"}` | Another iPhone already claimed the slot | `rm data/trusted_iphone.json` on the Mac |
| `{"ok": false, "error": "rate_limited"}` | Ran the Shortcut twice in one second | Wait 1 second and retry |
| `{"ok": false, "error": "bad_state"}` | Typo in the presence state | Use `at_desk`, `leaving`, `home`, `away`, or `busy` |
| ATOM says "I need you to verify on your iPhone first, Boss." | Face ID freshness expired (default 5 min) | Re-run **Verify with ATOM** |

Bridge audit log lives at `logs/atom_bridge_audit.jsonl` -- each
line is a single JSON object tagging the source IP, endpoint, and
reason. Tail it while debugging.

---

## 9. Related files

- [core/cross_device/iphone_bridge.py](../core/cross_device/iphone_bridge.py) -- the listener.
- [core/cross_device/bridge_auth.py](../core/cross_device/bridge_auth.py) -- token + audit.
- [core/cross_device/trusted_device.py](../core/cross_device/trusted_device.py) -- single-device lock.
- [core/identity_engine.py](../core/identity_engine.py) -- `is_owner_verified()` and Face ID freshness.
- [core/identity/device_binding.py](../core/identity/device_binding.py) -- `is_trusted_iphone()`.
- [docs/shortcuts/README.md](shortcuts/README.md) -- pre-canned action recipes you can import.

---

## 10. Tailscale + Enchanted (Sprint P4.4, Apr 26 2026)

The Phase 1 setup above gives you triggers from the iPhone Shortcuts
app. Phase 2 (P4.4 in [docs/ATOM_NEXT_STEPS_PLAN.md](ATOM_NEXT_STEPS_PLAN.md)) wires
ATOM's local brain into a real **conversational** iPhone client over
[Tailscale](https://tailscale.com/) — no public ports, no Apple
developer account, no pay-walled cloud.

The bridge now ships an **OpenAI-compatible `/v1/*` shim** so any
mainstream iOS LLM client (Enchanted, ChatX, GPTMobile-fork, etc.)
will work. We use [Enchanted](https://github.com/AugustDev/enchanted)
as the reference client because it's free, open source, and explicitly
supports an "Ollama" / OpenAI-style backend URL.

End-to-end stack:

```
iPhone 15
  └─ Enchanted (App Store, free)
       │  HTTPS-over-WireGuard (Tailscale tunnel; encrypted, no NAT punching)
       ▼
MacBook Air M5 (server)
  └─ ATOM bridge on 100.x.x.x:8787 (cross_device.bridge_port)
       │  Tailscale exposes the port to your tailnet only (private subnet)
       ▼
ATOM's MLX brain (Qwen3-8B-4bit, local, ANE-accelerated)
```

What you get:

- A real chat with ATOM's local brain from anywhere — coffee shop,
  commute, kitchen — over an encrypted WireGuard tunnel without
  exposing a single port to the open internet.
- Same auth model as the Shortcuts bridge (one shared token in iCloud
  Keychain). Enchanted lets you paste it as the API key.
- Streamed replies (SSE) so first-token latency on iPhone matches the
  Mac dashboard.

### 10.1 Prerequisites

1. A free Tailscale account on the same email/Apple ID for both Mac
   and iPhone.
2. Enchanted installed on iPhone:
   [App Store link](https://apps.apple.com/app/enchanted-llm/id6474268307).
3. ATOM running on the Mac with `cross_device.enabled: true` and
   the bridge port reachable on **localhost** (already covered in §0).

### 10.2 Install Tailscale on the Mac

```bash
# Homebrew is the cleanest path on macOS 26.x:
brew install --cask tailscale

# Or grab the pkg from https://tailscale.com/download/macos
```

Open Tailscale, sign in. The menubar icon shows a green dot when
the tunnel is up. Note your Mac's tailnet IP — it looks like
`100.64.0.5` and is **stable** (Tailscale assigns it once).

```bash
# From the Mac, confirm:
tailscale ip -4
# 100.64.0.5
```

### 10.3 Install Tailscale on the iPhone

1. App Store → search **Tailscale** → install.
2. Open it, sign in with the same account.
3. Toggle **Connect**. The iPhone gets its own `100.x.x.x` IP.

### 10.4 Make the bridge listen on the tailnet (NOT 0.0.0.0)

Edit `config/settings.json`:

```json
"cross_device": {
    "enabled": true,
    "bridge_port": 8787,
    "bind_host": "0.0.0.0",
    "allow_origins": ["100.64.0.0/10", "127.0.0.1"],
    "faceid_freshness_s": 300,
    ...
}
```

> **Why `0.0.0.0` is safe here:** Tailscale's WireGuard tunnel
> means the only IPs that can reach the Mac on this port are
> machines you've explicitly added to your tailnet. The
> `allow_origins` list is a *belt-and-suspenders* CIDR check ATOM
> applies on top.
>
> If you have **any** doubt about the network you're on (public
> Wi-Fi, hotel, conference) keep `bind_host: "127.0.0.1"` and only
> flip to `0.0.0.0` when you're back on a trusted Tailscale-only
> path. Tailscale's `--shields-up` mode is also worth enabling on
> the Mac if it's ever shared.

After editing, restart ATOM. The boot banner now shows the bridge
URL on the tailnet IP:

```
│  iPhone bridge ONLINE  ->  http://100.64.0.5:8787
│  Token file: config/bridge_token
```

### 10.5 Smoke-test from the iPhone before involving Enchanted

In Safari on the iPhone:

```
http://100.64.0.5:8787/health
```

You should see `{"ok": true, "version": 1}`. If you get
`Could not connect`, your tailnet isn't routing — re-check both
sides have Tailscale toggled on.

### 10.6 Configure Enchanted

In Enchanted on iPhone:

1. Tap the gear icon → **Servers**.
2. Add a server:
   - **Type:** `OpenAI-compatible`
   - **URL:** `http://100.64.0.5:8787/v1`  *(use **your** tailnet IP)*
   - **API Key:** paste the contents of `config/bridge_token`
     from the Mac. iCloud Keychain syncs it from §1 if you stored
     it there earlier — you can long-press the API Key field and
     paste from the keychain entry named `atom-bridge`.
3. Tap **Save** then **Models** — Enchanted should fetch `atom-local`
   from `/v1/models`. If you see an error, the most common cause
   is a wrong token (compare `cat config/bridge_token` to what you
   pasted).
4. Start a new chat. Type "hello, who are you?" — you should see
   ATOM's reply stream in token-by-token, identical to the Mac
   dashboard.

### 10.7 What ATOM exposes via `/v1/*`

The shim is intentionally minimal and **does not** invoke ATOM's
voice state machine, persona-pin, or full prompt-builder. It is a
pure "ask the local brain over Tailscale" surface. That means:

- **No tool execution.** iPhone-initiated chats cannot run shell
  commands, open apps, or trigger actions. Use the
  `/trigger` endpoint (Shortcut 3 in §4) for action commands.
- **No context from the Mac session.** The shim only sees the
  messages in the request body. Conversation history is whatever
  Enchanted sends — independent of what's on the Mac dashboard.
- **No persona file applied.** The shim sends raw ChatML through
  the model's tokenizer template; if you want ATOM-style replies
  on iPhone, paste a one-line system prompt into Enchanted's
  per-server "system message" field.

| Endpoint | Method | Auth | Behaviour |
|---|---|---|---|
| `/v1/models` | GET | `X-ATOM-Token` or `Authorization: Bearer <token>` | Lists `atom-local` |
| `/v1/chat/completions` | POST `stream=false` | same | Returns full reply once generated |
| `/v1/chat/completions` | POST `stream=true` | same | SSE: opener delta → content deltas → `data: [DONE]` |

### 10.8 Verify the whole loop with the smoke test

From the Mac (with ATOM running):

```bash
python scripts/iphone_bridge_smoke.py \
    --base-url http://127.0.0.1:8787 \
    --token "$(cat config/bridge_token)"
```

Expected: all three checks pass (models / non-stream / stream) and
total roundtrip < 5s warm. From an iPhone running Enchanted, the
same `python scripts/iphone_bridge_smoke.py` against the tailnet IP
proves the tunnel itself isn't the bottleneck.

### 10.9 ACL hardening (read this twice)

- **Don't add `0.0.0.0` to `allow_origins`** — that's a different
  setting (CIDR allowlist) and adding `0.0.0.0` would bypass the
  belt-and-suspenders check. Use `100.64.0.0/10` (Tailscale's
  CGNAT range) or your specific iPhone tailnet IP.
- **Keep `cross_device.faceid_freshness_s = 300`** so iPhone-driven
  tier-3 actions still need a fresh Face ID confirmation every 5
  minutes.
- **Audit the bridge log weekly:**

   ```bash
   tail -100 logs/atom_bridge_audit.jsonl | jq .
   ```

   Lines only appear on auth failures (bad/missing token, device
   conflict). If you see lines from an IP that isn't in your
   tailnet, **rotate the token immediately** by deleting
   `config/bridge_token` and restarting ATOM.

### 10.10 Troubleshooting (Tailscale-specific)

| Symptom | Cause | Fix |
|---|---|---|
| `Could not connect` from Safari at `:8787` | Tailscale not connected on either side | Toggle Tailscale on both Mac and iPhone; check menubar/iOS toggle |
| `Could not connect` only from iPhone (Safari OK on Mac) | `bind_host: 127.0.0.1` blocks tailnet IP | Set `bind_host: 0.0.0.0` and restart |
| Enchanted says "Failed to fetch models" | Token mismatch | Re-paste `config/bridge_token` into Enchanted; some keyboards mangle `_` and `-` |
| Replies cut off mid-sentence | Default `max_tokens=512` ran out | Tap the model in Enchanted → Advanced → bump `max_tokens` |
| First token takes >5s | Cold MLX brain (model not loaded yet) | Run one chat from the Mac first to warm; subsequent iPhone calls are warm |
| `429 rate_limited` | Two rapid requests from same IP | Wait 0.5s and retry; this is the per-source-IP soft limit |
