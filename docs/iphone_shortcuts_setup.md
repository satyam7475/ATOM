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
