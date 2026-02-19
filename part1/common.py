"""
Common utilities for the tokenizer.
"""

from pathlib import Path

# Path to test fixtures
FIXTURES_PATH = Path(__file__).parent / "fixtures"


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping from bytes to unicode characters used by GPT-2.

    GPT-2 uses a reversible mapping from bytes to unicode strings. This avoids
    mapping bytes to whitespace/control characters which can cause issues.

    The mapping works as follows:
    - Printable ASCII characters (! through ~, and space) map to themselves
    - Other bytes are mapped to unicode characters starting at U+0100

    Returns:
        Dictionary mapping byte values (0-255) to unicode characters
    """
    # These are the "good" bytes that don't need remapping
    # They're printable ASCII characters
    bs = (
        list(range(ord("!"), ord("~") + 1))  # ! to ~
        + list(range(ord("¡"), ord("¬") + 1))  # ¡ to ¬
        + list(range(ord("®"), ord("ÿ") + 1))  # ® to ÿ
    )
    cs = bs[:]

    # For all other bytes, we map them to unicode chars starting at 256
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
