# macOS Focus / Do-Not-Disturb -- Apple Shortcuts Setup

ATOM controls Focus through three Apple Shortcuts that **you create
once** in the macOS Shortcuts app. After that, voice commands like
"focus mode on", "do not disturb for 30 minutes", and "is focus on"
work end-to-end.

> Why Shortcuts and not the old `defaults` plist? Apple removed the
> public preference key on macOS Ventura+. Shortcuts is the only
> first-party API that flips Focus reliably and silently from the
> command line.

## Required shortcuts

| Shortcut name | Purpose |
| --- | --- |
| `ATOM Focus On` | Turn on Do Not Disturb (or any Focus you prefer). |
| `ATOM Focus Off` | Turn off Do Not Disturb. |
| `ATOM Focus Status` | *(Optional)* Echo `on` or `off` so ATOM can answer "is focus on". |

The names are case-insensitive but the spelling must match exactly.

## Quick build (3 minutes)

1. Open the **Shortcuts** app on macOS (Spotlight: `Shortcuts`).
2. Click `+` to create a new shortcut.
3. Search the action library for **Set Focus** and drag it into the
   shortcut.
   - Configure the action: `Turn` -> `On` -> `Do Not Disturb`.
4. Name the shortcut **ATOM Focus On**, then save.
5. Repeat steps 2-4 for **ATOM Focus Off**, but set the **Set Focus**
   action to `Turn` -> `Off`.

### Adding a duration (optional, recommended)

To support "do not disturb for 30 minutes":

1. In **ATOM Focus On**, add a **Get Shortcut Input** action at the
   top.
2. Add an **If** action: `If Shortcut Input has any value`.
   - Inside the `If`: drag in **Set Focus** -> `Turn On` and add
     `Until` -> `Custom` -> use the `Shortcut Input` magic variable
     for minutes.
   - In the `Otherwise`: drag in **Set Focus** -> `Turn On` (no
     duration).

ATOM passes the integer minute count as the shortcut input
automatically.

### Status shortcut (optional)

To support "is focus on":

1. Create **ATOM Focus Status**.
2. Add **Get Current Focus**.
3. Add an **If** action: `If Current Focus has any value`.
   - Inside: **Text** action with the literal `on`, then **Output**.
   - Otherwise: **Text** action with `off`, then **Output**.

## Verifying

From the terminal:

```bash
shortcuts list | grep "ATOM Focus"
shortcuts run "ATOM Focus On"
shortcuts run "ATOM Focus Off"
shortcuts run "ATOM Focus Status"
```

If you see all three shortcuts listed and they each finish without
error, ATOM is ready. Try a voice command:

- "focus mode on"
- "do not disturb for forty-five minutes"
- "is focus on"
- "focus off"

## Troubleshooting

- **"I need an Apple Shortcut named '...'"** — The shortcut is
  missing or misspelled. Re-check the name in the Shortcuts app.
- **`shortcuts: command not found`** — The CLI ships with
  Xcode's command-line tools. Run `xcode-select --install`.
- **The shortcut runs but nothing changes** — Open *System Settings ->
  Notifications & Focus -> Focus*, ensure "Do Not Disturb" exists,
  then re-open the shortcut and confirm it points at "Do Not Disturb"
  (not at a different Focus profile).
- **Permission prompt every time** — Open *System Settings ->
  Privacy & Security -> Automation*, then allow the Shortcuts app to
  control "System Events" once.

## Why this is safer than UI scripting

Earlier ATOM builds tried to toggle DND through `osascript` UI tap
sequences. That broke whenever Apple repositioned the menubar widget.
Shortcuts is a stable, sandbox-friendly API — no Accessibility prompts
beyond the initial permission, no menubar fragility, and the same
script keeps working across macOS major versions.
