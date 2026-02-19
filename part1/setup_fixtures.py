#!/usr/bin/env python3
"""
Setup script to generate the GPT-2 vocab and merges from tiktoken.
Run this once before running the tests.
"""

import json
import os
from pathlib import Path


def bytes_to_unicode():
    """GPT-2's byte to unicode mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def setup_fixtures():
    """Generate GPT-2 vocab and merges using tiktoken."""
    import tiktoken

    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    # Get the GPT-2 encoding
    enc = tiktoken.get_encoding("gpt2")
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    # Build vocab: token_string -> token_id
    # tiktoken's _mergeable_ranks gives us token_bytes -> rank
    vocab = {}
    for token_bytes, token_id in enc._mergeable_ranks.items():
        # Convert bytes to GPT-2's unicode representation
        token_str = "".join(byte_encoder[b] for b in token_bytes)
        vocab[token_str] = token_id

    # Add special tokens
    for token_str, token_id in enc._special_tokens.items():
        vocab[token_str] = token_id

    # Save vocab
    vocab_path = fixtures_dir / "gpt2_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Saved vocab with {len(vocab)} tokens to {vocab_path}")

    # For merges, we need to reconstruct them from the BPE ranks
    # The merges are ordered by rank (lower = earlier merge)
    # We need pairs of tokens that when merged give a token in vocab

    # Build merges from tiktoken's internal data
    # tiktoken doesn't expose merges directly, so we reconstruct them
    merges = []

    # Get all tokens sorted by ID (which corresponds to merge order for multi-byte tokens)
    tokens_by_id = sorted(enc._mergeable_ranks.items(), key=lambda x: x[1])

    # First 256 tokens are single bytes, no merges needed
    # For tokens after that, we need to find the merge that created them
    for token_bytes, token_id in tokens_by_id:
        if len(token_bytes) > 1:
            # Find the split point that gives two valid tokens
            for split_idx in range(1, len(token_bytes)):
                left = token_bytes[:split_idx]
                right = token_bytes[split_idx:]
                if left in enc._mergeable_ranks and right in enc._mergeable_ranks:
                    # Both parts exist as tokens, this could be the merge
                    left_str = "".join(byte_encoder[b] for b in left)
                    right_str = "".join(byte_encoder[b] for b in right)
                    merges.append(f"{left_str} {right_str}")
                    break

    # Save merges
    merges_path = fixtures_dir / "gpt2_merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for merge in merges:
            f.write(merge + "\n")
    print(f"Saved {len(merges)} merges to {merges_path}")

    # Create sample test files
    create_test_files(fixtures_dir)

    print("Setup complete!")


def create_test_files(fixtures_dir):
    """Create sample test files for the tests."""

    # address.txt - sample address
    address_content = """John Smith
    123 Main Street
    Anytown, CA 12345
    United States
    """
    (fixtures_dir / "address.txt").write_text(address_content)

    # german.txt - sample German text
    german_content = """Guten Tag! Wie geht es Ihnen?
Ich heiße Max und komme aus München.
Das Wetter ist heute sehr schön."""
    (fixtures_dir / "german.txt").write_text(german_content)

    # tinystories_sample.txt
    tinystories_content = """Once upon a time, there was a little girl named Lily. She loved to play in the garden with her dog, Max. One sunny day, Lily found a beautiful butterfly.

    "Look, Max!" she said. "Isn't it pretty?"

    The butterfly flew away, and Lily chased it through the flowers. She laughed and played until the sun went down.

    <|endoftext|>

    Tom was a curious boy who loved to explore. One day, he found an old map in his grandfather's attic.

    "What's this?" he wondered.

    The map showed a path to a hidden treasure in the woods behind his house."""
    (fixtures_dir / "tinystories_sample.txt").write_text(tinystories_content)

    # tinystories_sample_5M.txt - larger file for memory tests
    large_content = tinystories_content * 100  # Repeat to make it larger
    (fixtures_dir / "tinystories_sample_5M.txt").write_text(large_content)

    # special_token_trailing_newlines.txt
    special_newlines = """Hello world<|endoftext|>

"""
    (fixtures_dir / "special_token_trailing_newlines.txt").write_text(special_newlines)

    # special_token_double_newlines_non_whitespace.txt
    special_double = """First part<|endoftext|>

Second part with text"""
    (fixtures_dir / "special_token_double_newlines_non_whitespace.txt").write_text(
        special_double
    )

    print("Created test fixture files")


if __name__ == "__main__":
    setup_fixtures()
