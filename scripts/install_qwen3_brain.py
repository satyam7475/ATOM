"""Idempotent installer for ATOM's primary brain (Qwen3-4B-Instruct-2507-4bit).

Why this script exists
----------------------
The mlx-community quantized release of Qwen3-4B-Instruct-2507-4bit ships
a tokenizer_config.json that omits ``chat_template``. Loading the
tokenizer via ``mlx_lm.load`` therefore returns a ``chat_template = None``
tokenizer, which silently breaks every code path that calls
``tokenizer.apply_chat_template(...)`` — including the live brain smoke
test in ``tests/test_brain_qwen_smoke.py`` and any future caller that
wants chat-formatted prompts (Qwen2.5 used to have it embedded; Qwen3
does not).

ATOM's runtime works around this today by feeding raw prompts directly
to ``stream_generate`` (see ``brain/mlx_llm.py``). But that's brittle:
the moment any downstream code wants ChatML formatting we'd silently
drop the system role / generation prefix and produce malformed prompts.

This script makes the install reproducible:
  1. Snapshot-download the model if the directory is missing.
  2. Patch the local ``tokenizer_config.json`` to embed the canonical
     chat template fetched from the upstream FP16 repo
     ``Qwen/Qwen3-4B-Instruct-2507`` (the original, non-quantized).
  3. Verify ``tokenizer.apply_chat_template`` produces the expected
     ChatML output with ``<|im_start|>``/``<|im_end|>`` markers.

Idempotent. Safe to re-run after a model upgrade.

Usage:
    python scripts/install_qwen3_brain.py
    python scripts/install_qwen3_brain.py --force-template-refresh

Owner: Satyam
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set BEFORE any HF tokenizer touches the process to avoid
# semaphore leaks at shutdown (same fix as main.py).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

QUANT_REPO = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
TEMPLATE_REPO = "Qwen/Qwen3-4B-Instruct-2507"
LOCAL_DIR = ROOT / "models" / "qwen3-4b-instruct-4bit"
TOKENIZER_CONFIG = LOCAL_DIR / "tokenizer_config.json"


def _has_template(tok_cfg: dict) -> bool:
    val = tok_cfg.get("chat_template")
    return isinstance(val, str) and len(val) > 100


def _pull_model() -> None:
    if LOCAL_DIR.exists() and any(LOCAL_DIR.glob("*.safetensors")):
        print(f"[ok] model already present at {LOCAL_DIR}")
        return
    from huggingface_hub import snapshot_download

    print(f"[pull] {QUANT_REPO} -> {LOCAL_DIR}")
    t0 = time.monotonic()
    snapshot_download(
        repo_id=QUANT_REPO,
        local_dir=str(LOCAL_DIR),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"],
    )
    dt = time.monotonic() - t0
    total = sum(p.stat().st_size for p in LOCAL_DIR.rglob("*") if p.is_file())
    print(f"[pull] complete in {dt:.1f}s, {total/1e9:.2f} GB on disk")


def _patch_template(*, force_refresh: bool) -> bool:
    if not TOKENIZER_CONFIG.exists():
        raise FileNotFoundError(
            f"tokenizer_config.json missing at {TOKENIZER_CONFIG} -- "
            f"model pull may have failed",
        )
    cfg = json.loads(TOKENIZER_CONFIG.read_text())
    if _has_template(cfg) and not force_refresh:
        print("[ok] chat_template already embedded; skipping refresh "
              "(use --force-template-refresh to override)")
        return False

    from huggingface_hub import hf_hub_download

    print(f"[patch] fetching canonical chat_template from {TEMPLATE_REPO}")
    upstream_path = hf_hub_download(
        repo_id=TEMPLATE_REPO, filename="tokenizer_config.json",
    )
    upstream_cfg = json.loads(Path(upstream_path).read_text())
    template = upstream_cfg.get("chat_template")
    if not isinstance(template, str) or len(template) < 100:
        raise RuntimeError(
            f"Upstream {TEMPLATE_REPO} no longer has a chat_template! "
            f"This script needs updating.",
        )

    cfg["chat_template"] = template
    TOKENIZER_CONFIG.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
    )
    print(f"[patch] embedded chat_template ({len(template)} chars) "
          f"into {TOKENIZER_CONFIG}")
    return True


def _verify() -> None:
    print("[verify] loading via mlx_lm.load...")
    from mlx_lm import load

    _model, tokenizer = load(str(LOCAL_DIR))
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "tokenizer.chat_template is still empty after patch -- mlx_lm "
            "may be reading from a different config",
        )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are ATOM, Satyam's personal OS."},
            {"role": "user", "content": "Self-check: are you online?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if "<|im_start|>" not in rendered or "<|im_end|>" not in rendered:
        raise RuntimeError(
            f"Chat template rendered without ChatML markers: {rendered[:200]!r}",
        )
    if "<|im_start|>assistant" not in rendered:
        raise RuntimeError(
            "Chat template missing assistant generation prefix",
        )
    print("[verify] tokenizer.apply_chat_template produces canonical ChatML")
    print("[verify] sample render:")
    for line in rendered.splitlines():
        print(f"  | {line}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force-template-refresh", action="store_true",
        help="Re-fetch the chat_template from upstream even if one is "
        "already embedded (use after a Qwen3 template revision)",
    )
    args = p.parse_args()

    print("== ATOM brain installer ==")
    print(f"  target: {LOCAL_DIR}")
    print(f"  quantized weights: {QUANT_REPO}")
    print(f"  template source: {TEMPLATE_REPO}")
    print()

    _pull_model()
    _patch_template(force_refresh=args.force_template_refresh)
    _verify()
    print()
    print("== ATOM brain ready ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
