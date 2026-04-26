"""Idempotent installer for ATOM's primary brain (Qwen3 MLX 4-bit).

Variants
--------
* ``--variant 4b`` (default): ``mlx-community/Qwen3-4B-Instruct-2507-4bit``
  -- the original ATOM brain. The upstream quantized tokenizer_config
  omits ``chat_template`` so this script patches it from
  ``Qwen/Qwen3-4B-Instruct-2507``. Required for callers using
  ``tokenizer.apply_chat_template(...)`` to produce ChatML.
* ``--variant 8b`` (Sprint Ω): ``mlx-community/Qwen3-8B-4bit``
  -- the bigger brain. Upstream ships the chat_template embedded so
  no patch is needed; the patch step is skipped automatically.
* ``--variant 0.6b`` (Sprint Ω.10): ``mlx-community/Qwen3-0.6B-4bit``
  -- the *ultra* brain (also serves as the speculative-decoding draft
  for the 4B target). Upstream's quantized tokenizer_config also
  omits ``chat_template``, so this script patches it from
  ``Qwen/Qwen3-0.6B``. Critical: the draft tokenizer's vocabulary
  must match the 4B target — the runtime double-checks this at
  load time.

Why the patch exists
--------------------
The mlx-community 4-bit Qwen3-4B-Instruct-2507 release omits the
``chat_template`` from its tokenizer_config.json. Loading via
``mlx_lm.load`` therefore returns a ``chat_template = None`` tokenizer,
silently breaking every caller of ``tokenizer.apply_chat_template(...)``
-- including the brain smoke test and any code that wants ChatML
formatting. ATOM's runtime works around this by feeding raw prompts
to ``stream_generate``, but that's brittle. This script makes the
install reproducible by patching the template idempotently.

Idempotent. Safe to re-run after a model upgrade.

Usage:
    python scripts/install_qwen3_brain.py                  # 4b (default)
    python scripts/install_qwen3_brain.py --variant 8b     # 8b (Sprint Ω)
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


# ── variant catalogue ────────────────────────────────────────────────
# Keys: short variant name. Values: install metadata.
#   quant_repo:    HF repo containing the MLX 4-bit weights
#   template_repo: optional HF repo whose tokenizer_config carries the
#                  canonical chat_template. Set to None when the quant
#                  repo already ships the template (no patch needed).
#   local_dirname: subdirectory under ./models for the local copy
_VARIANTS: dict[str, dict[str, object]] = {
    "4b": {
        "quant_repo": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "template_repo": "Qwen/Qwen3-4B-Instruct-2507",
        "local_dirname": "qwen3-4b-instruct-4bit",
    },
    "8b": {
        "quant_repo": "mlx-community/Qwen3-8B-4bit",
        "template_repo": None,
        "local_dirname": "qwen3-8b-4bit",
    },
    "0.6b": {
        "quant_repo": "mlx-community/Qwen3-0.6B-4bit",
        "template_repo": "Qwen/Qwen3-0.6B",
        "local_dirname": "qwen3-0.6b-instruct-4bit",
    },
}

# Module-level constants kept for backward compatibility with any caller
# that imports them directly. They reflect the default (4B) variant.
QUANT_REPO = str(_VARIANTS["4b"]["quant_repo"])
TEMPLATE_REPO = str(_VARIANTS["4b"]["template_repo"])
LOCAL_DIR = ROOT / "models" / str(_VARIANTS["4b"]["local_dirname"])
TOKENIZER_CONFIG = LOCAL_DIR / "tokenizer_config.json"


def _has_template(tok_cfg: dict) -> bool:
    val = tok_cfg.get("chat_template")
    return isinstance(val, str) and len(val) > 100


def _pull_model(quant_repo: str, local_dir: Path) -> None:
    if local_dir.exists() and any(local_dir.glob("*.safetensors")):
        print(f"[ok] model already present at {local_dir}")
        return
    from huggingface_hub import snapshot_download

    print(f"[pull] {quant_repo} -> {local_dir}")
    t0 = time.monotonic()
    snapshot_download(
        repo_id=quant_repo,
        local_dir=str(local_dir),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"],
    )
    dt = time.monotonic() - t0
    total = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file())
    print(f"[pull] complete in {dt:.1f}s, {total/1e9:.2f} GB on disk")


def _patch_template(
    tokenizer_config: Path,
    template_repo: str | None,
    *,
    force_refresh: bool,
) -> bool:
    """Embed the canonical chat_template if it's missing.

    When ``template_repo`` is ``None`` (e.g. the 8B variant whose quant
    repo already ships the template), the patch is skipped after a
    sanity check. ``force_refresh`` only re-fetches when there IS a
    canonical source repo to fetch from.
    """
    if not tokenizer_config.exists():
        raise FileNotFoundError(
            f"tokenizer_config.json missing at {tokenizer_config} -- "
            f"model pull may have failed",
        )
    cfg = json.loads(tokenizer_config.read_text())
    has_template = _has_template(cfg)

    if template_repo is None:
        if not has_template:
            raise RuntimeError(
                f"{tokenizer_config} has no chat_template and no "
                f"template source repo is configured for this variant. "
                f"Either pick a different variant or set template_repo "
                f"explicitly.",
            )
        print(
            "[ok] chat_template already embedded by the quant repo; "
            "no patch needed for this variant",
        )
        return False

    if has_template and not force_refresh:
        print("[ok] chat_template already embedded; skipping refresh "
              "(use --force-template-refresh to override)")
        return False

    from huggingface_hub import hf_hub_download

    print(f"[patch] fetching canonical chat_template from {template_repo}")
    upstream_path = hf_hub_download(
        repo_id=template_repo, filename="tokenizer_config.json",
    )
    upstream_cfg = json.loads(Path(upstream_path).read_text())
    template = upstream_cfg.get("chat_template")
    if not isinstance(template, str) or len(template) < 100:
        raise RuntimeError(
            f"Upstream {template_repo} no longer has a chat_template! "
            f"This script needs updating.",
        )

    cfg["chat_template"] = template
    tokenizer_config.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
    )
    print(f"[patch] embedded chat_template ({len(template)} chars) "
          f"into {tokenizer_config}")
    return True


def _verify(local_dir: Path) -> None:
    print("[verify] loading via mlx_lm.load...")
    from mlx_lm import load

    _model, tokenizer = load(str(local_dir))
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
        "--variant",
        choices=sorted(_VARIANTS.keys()),
        default="4b",
        help="Which Qwen3 size to install (default: 4b).",
    )
    p.add_argument(
        "--force-template-refresh", action="store_true",
        help="Re-fetch the chat_template from upstream even if one is "
        "already embedded (use after a Qwen3 template revision). "
        "No-op for variants that have no canonical source repo.",
    )
    args = p.parse_args()

    spec = _VARIANTS[args.variant]
    quant_repo = str(spec["quant_repo"])
    template_repo = spec["template_repo"]  # type: ignore[assignment]
    if template_repo is not None:
        template_repo = str(template_repo)
    local_dir = ROOT / "models" / str(spec["local_dirname"])
    tokenizer_config = local_dir / "tokenizer_config.json"

    print("== ATOM brain installer ==")
    print(f"  variant: {args.variant}")
    print(f"  target: {local_dir}")
    print(f"  quantized weights: {quant_repo}")
    print(f"  template source: {template_repo or '(embedded -- no patch needed)'}")
    print()

    _pull_model(quant_repo, local_dir)
    _patch_template(
        tokenizer_config,
        template_repo,
        force_refresh=args.force_template_refresh,
    )
    _verify(local_dir)
    print()
    print(f"== ATOM brain ready ({args.variant}) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
