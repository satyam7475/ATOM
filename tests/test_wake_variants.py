from voice.listening_modes import WakeWordFilter


def test_atom_wake_variants_include_common_stt_mishears() -> None:
    for phrase in (
        "hey adam",
        "hey adom",
        "hey adtan",
        "hey atan",
        "ok adton are you there",
    ):
        assert WakeWordFilter.contains_wake(phrase), phrase


def test_wake_filter_detects_adtan_partial() -> None:
    wake = WakeWordFilter(cooldown_s=0.3)

    assert wake.check("hey adtan can you hear me") == "hey adtan"
