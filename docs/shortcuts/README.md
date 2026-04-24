# iPhone Shortcut recipes for ATOM

Apple's `.shortcut` format is a binary plist signed with your Apple
ID; we cannot ship a real one in the repo (Apple won't accept a
signed-by-another-account file on your iPhone, and unsigned ones
are rejected). So what lives here is the next best thing: a
**JSON recipe per shortcut** listing every action in order. Follow
it step-by-step in the stock Shortcuts app and you'll end up with
a functionally identical shortcut in ~2 minutes per recipe.

See [../iphone_shortcuts_setup.md](../iphone_shortcuts_setup.md) for
the full setup flow (token storage, Mac prerequisites, troubleshooting).

## Recipes

| File | Shortcut name | Purpose |
|---|---|---|
| [verify_with_atom.recipe.json](verify_with_atom.recipe.json) | Verify with ATOM | Face ID -> POST `/faceid`. Unlocks tier-3 tools for 5 min. |
| [im_at_desk.recipe.json](im_at_desk.recipe.json) | I'm at desk | POST `/presence` state=at_desk. Triggers greeting + briefing. |
| [im_leaving.recipe.json](im_leaving.recipe.json) | I'm leaving | POST `/presence` state=leaving. Puts ATOM in away mode. |
| [morning_routine.recipe.json](morning_routine.recipe.json) | Run morning routine | POST `/trigger` name=morning_routine. Fires the briefing. |

## Recipe file format

Every recipe is a JSON file with two top-level keys:

```json
{
  "name": "Verify with ATOM",
  "actions": [
    {"action": "Get Device Details", "param": "Name", "store_as": "device_id"},
    ...
  ]
}
```

Each action entry maps 1:1 to a Shortcuts app action. The
`param` is the sub-option you pick in the UI; `store_as` is the
Magic Variable name to set.

Recipes reference two external things you configure once in the
Shortcuts app before running any of them:

- `MAC_LAN_IP` -- your Mac's LAN IP (find via `ipconfig getifaddr en0`).
  Set this in a Shortcut **variable** or just paste it verbatim in
  each URL action.
- The iCloud Keychain account **`atom-bridge` / `atom`** holding the
  bridge token minted at `config/bridge_token`.

If you change the Mac's IP or the port (bridge chose 8788 because
8787 was busy? see `logs/atom_bridge.port`), update the URL in each
shortcut -- or wrap the host+port in another Shortcut called
`_atom_base_url` that you call from each.

## Importing hints

A future Phase 1.5 could ship a one-tap **Import to Shortcuts** link
using iCloud Drive sharing. For now the manual recipe walkthrough
in [../iphone_shortcuts_setup.md](../iphone_shortcuts_setup.md)
is the supported path.
