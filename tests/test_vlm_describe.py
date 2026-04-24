"""Tests for the optional VLM captioner (SmolVLM by default).

These tests never import ``mlx-vlm`` — they either exercise the
fail-open paths (model dir missing, import missing) or monkey-patch
the module-level ``_mlx_vlm_generate_caption`` adapter so the happy
path can be verified without downloading a ~1.4 GB model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.perception import vlm_describe
from core.perception.vlm_describe import VLMCaptioner, _tidy_caption


def test_disabled_when_model_path_missing(tmp_path):
    cap = VLMCaptioner(model_path=tmp_path / "does-not-exist")
    assert cap.is_available is False
    reason = cap.disabled_reason()
    assert "VLM weights not found" in reason
    # Fetch hint must be copy-pasteable
    assert "hf download" in reason
    # describe() is safe to call and returns "" without raising
    assert cap.describe("/tmp/any.jpg") == ""


def test_repo_fallback_makes_captioner_available_without_local_dir(tmp_path):
    cap = VLMCaptioner(
        model_path=tmp_path / "does-not-exist",
        model_repo="mlx-community/SmolVLM-Instruct-4bit",
    )
    # Repo-id fallback: the captioner should *not* declare itself
    # offline just because the local dir is missing — mlx-vlm.load
    # will fetch on first describe() call.
    assert cap.is_available is True
    assert cap.disabled_reason() == ""


def test_describe_returns_empty_when_jpeg_missing(tmp_path):
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    cap = VLMCaptioner(model_path=model_dir)
    # Even with a valid-looking model dir, describe on a missing
    # image must degrade to "" with a debug log, not raise.
    missing_jpeg = tmp_path / "no-such-frame.jpg"
    assert cap.describe(missing_jpeg) == ""


def test_import_failure_is_fail_open(monkeypatch, tmp_path):
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    # Stub a single JPEG so the path check in describe() passes.
    jpeg = tmp_path / "frame.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xd9")

    def _boom_import(self):  # noqa: ARG001
        # Simulate mlx-vlm not being installed. _load runs the import
        # inline — we patch the method to raise on its behalf.
        self._load_failed = True
        self._load_error = "mlx-vlm is not installed"
        return False

    monkeypatch.setattr(VLMCaptioner, "_load", _boom_import)
    cap = VLMCaptioner(model_path=model_dir)
    result = cap.describe(jpeg)
    assert result == ""
    # Subsequent calls remain no-ops (load-failed latch)
    assert cap.describe(jpeg) == ""


def test_describe_happy_path_tidies_caption(monkeypatch, tmp_path):
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    jpeg = tmp_path / "frame.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xd9")

    # Pretend the VLM loaded successfully.
    def _fake_load(self):
        self._model = object()
        self._processor = object()
        return True

    captured_prompts: list[str] = []

    def _fake_generate(*, model, processor, config, image_path, prompt, max_tokens, temperature):  # noqa: ARG001
        captured_prompts.append(prompt)
        return "  Caption: A laptop on a wooden desk.   \n"

    monkeypatch.setattr(VLMCaptioner, "_load", _fake_load)
    monkeypatch.setattr(
        vlm_describe, "_mlx_vlm_generate_caption", _fake_generate,
    )

    cap = VLMCaptioner(
        model_path=model_dir,
        prompt="Describe this image in one short sentence.",
        max_tokens=48,
    )
    result = cap.describe(jpeg)
    assert result == "A laptop on a wooden desk."
    assert captured_prompts == [
        "Describe this image in one short sentence.",
    ]


def test_describe_custom_prompt_and_max_tokens(monkeypatch, tmp_path):
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    jpeg = tmp_path / "frame.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xd9")

    def _fake_load(self):
        self._model = object()
        self._processor = object()
        return True

    seen: dict = {}

    def _fake_generate(*, model, processor, config, image_path, prompt, max_tokens, temperature):  # noqa: ARG001
        seen["prompt"] = prompt
        seen["max_tokens"] = max_tokens
        return "text in the image reads hello"

    monkeypatch.setattr(VLMCaptioner, "_load", _fake_load)
    monkeypatch.setattr(
        vlm_describe, "_mlx_vlm_generate_caption", _fake_generate,
    )

    cap = VLMCaptioner(model_path=model_dir)
    result = cap.describe(jpeg, prompt="what text is visible", max_tokens=12)
    assert result == "text in the image reads hello"
    assert seen["prompt"] == "what text is visible"
    assert seen["max_tokens"] == 12


def test_inference_error_is_fail_open(monkeypatch, tmp_path):
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    jpeg = tmp_path / "frame.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xd9")

    def _fake_load(self):
        self._model = object()
        self._processor = object()
        return True

    def _explode(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("simulated inference crash")

    monkeypatch.setattr(VLMCaptioner, "_load", _fake_load)
    monkeypatch.setattr(
        vlm_describe, "_mlx_vlm_generate_caption", _explode,
    )

    cap = VLMCaptioner(model_path=model_dir)
    # Must never propagate the RuntimeError upstream.
    assert cap.describe(jpeg) == ""


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  Caption: A dog.  ", "A dog."),
        ("Answer: hello world\n", "hello world"),
        ("Description:   two cats\t", "two cats"),
        ("plain output", "plain output"),
        ("", ""),
        ("  multi    space\ntext", "multi space text"),
    ],
)
def test_tidy_caption(raw, expected):
    assert _tidy_caption(raw) == expected


def test_tidy_caption_truncates_long_output():
    long_text = "a " * 300  # well over the 280-char cap
    tidied = _tidy_caption(long_text)
    assert len(tidied) <= 280
    assert tidied.endswith("...")


def test_metrics_default_shape_when_idle(tmp_path):
    cap = VLMCaptioner(model_path=tmp_path / "missing")
    m = cap.metrics()
    # Schema must be stable so the status snapshot / dashboard never
    # has to defensively ``.get(...)`` -- every key always present.
    expected_keys = {
        "model_path",
        "model_repo",
        "is_loaded",
        "is_available",
        "load_failed",
        "load_error",
        "load_ms",
        "inference_count",
        "inference_failed_count",
        "last_inference_ms",
        "last_inference_age_s",
        "avg_inference_ms",
    }
    assert expected_keys.issubset(m.keys())
    assert m["inference_count"] == 0
    assert m["last_inference_ms"] == 0.0
    assert m["last_inference_age_s"] == 0.0


def test_metrics_after_describe_pass_records_load_and_inference(monkeypatch, tmp_path):
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    jpeg = tmp_path / "frame.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xd9")

    def _fake_load(self):
        # Mimic the real _load: stash a load_ms reading too.
        self._model = object()
        self._processor = object()
        self._load_ms = 1234.5
        return True

    def _fake_generate(*, model, processor, config, image_path, prompt, max_tokens, temperature):  # noqa: ARG001
        return "A clear sentence."

    monkeypatch.setattr(VLMCaptioner, "_load", _fake_load)
    monkeypatch.setattr(
        vlm_describe, "_mlx_vlm_generate_caption", _fake_generate,
    )

    cap = VLMCaptioner(model_path=model_dir)
    cap.describe(jpeg)
    cap.describe(jpeg)

    m = cap.metrics()
    assert m["is_loaded"] is True
    assert m["load_ms"] == 1234.5
    assert m["inference_count"] == 2
    assert m["inference_failed_count"] == 0
    assert m["last_inference_ms"] >= 0.0
    assert m["last_inference_age_s"] >= 0.0


def test_metrics_records_inference_failures(monkeypatch, tmp_path):
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    jpeg = tmp_path / "frame.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xd9")

    def _fake_load(self):
        self._model = object()
        self._processor = object()
        return True

    def _explode(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(VLMCaptioner, "_load", _fake_load)
    monkeypatch.setattr(
        vlm_describe, "_mlx_vlm_generate_caption", _explode,
    )

    cap = VLMCaptioner(model_path=model_dir)
    cap.describe(jpeg)
    cap.describe(jpeg)

    m = cap.metrics()
    # Both calls counted as inference attempts; both also counted as failures.
    assert m["inference_count"] == 2
    assert m["inference_failed_count"] == 2
