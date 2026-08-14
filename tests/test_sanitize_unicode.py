from pi_ai.utils.sanitize_unicode import sanitize_surrogates


def test_preserves_plain_ascii_text():
    assert sanitize_surrogates("Hello World") == "Hello World"


def test_preserves_real_emoji_code_point():
    # A real emoji is a single Python code point (not a UTF-16 surrogate pair
    # once decoded), so it passes through untouched.
    assert sanitize_surrogates("Hello \U0001f648 World") == "Hello \U0001f648 World"


def test_preserves_literal_paired_surrogates():
    # A literal adjacent high+low surrogate pair (as could occur decoding
    # JS-originated UTF-16 data one code unit at a time) is preserved as-is,
    # not merged into the code point it would represent in UTF-16.
    paired = "\ud83d\ude48"
    assert sanitize_surrogates(f"Text {paired} here") == f"Text {paired} here"


def test_removes_unpaired_high_surrogate():
    unpaired_high = "\ud83d"
    assert sanitize_surrogates(f"Text {unpaired_high} here") == "Text  here"


def test_removes_unpaired_low_surrogate():
    unpaired_low = "\ude00"
    assert sanitize_surrogates(f"Text {unpaired_low} here") == "Text  here"


def test_removes_high_surrogate_followed_by_non_surrogate():
    unpaired_high = "\ud83d"
    assert sanitize_surrogates(f"{unpaired_high}A") == "A"


def test_removes_low_surrogate_preceded_by_non_surrogate():
    unpaired_low = "\udc00"
    assert sanitize_surrogates(f"A{unpaired_low}") == "A"


def test_removes_two_high_surrogates_in_a_row():
    # Two consecutive high surrogates: neither is followed by a valid low
    # surrogate, so both are unpaired and removed.
    two_highs = "\ud800\ud801"
    assert sanitize_surrogates(f"x{two_highs}y") == "xy"


def test_empty_string():
    assert sanitize_surrogates("") == ""


def test_multiple_unpaired_surrogates_mixed_with_valid_text():
    text = "a\ud800b\udc00c" + "\ud83d\ude00" + "d\udfffe"
    # "a" + (removed) + "b" + (removed) + "c" + (valid pair kept) + "d" + (removed) + "e"
    assert sanitize_surrogates(text) == "abc" + "\ud83d\ude00" + "de"
