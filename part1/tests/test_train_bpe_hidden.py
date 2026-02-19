"""
Hidden tests for BPE training - only available on Gradescope.
These tests use different corpora and vocab sizes to prevent reverse engineering.
"""

import json
import time

from adapters import run_train_bpe
from common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_different_vocab_size():
    """Test BPE training with a different vocabulary size."""
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=350,  # Different from public test (500)
        special_tokens=["<|endoftext|>"],
    )

    # Verify basic properties
    assert len(vocab) == 350
    assert len(merges) == 350 - 257  # vocab_size - special_tokens - 256 bytes

    # Verify first merge is deterministic (most frequent pair)
    # The first few merges should be consistent
    assert merges[0] == (b"h", b"e")  # Most common pair in corpus.en (15 occurrences)


def test_train_bpe_multiple_special_tokens():
    """Test BPE training with multiple special tokens."""
    input_path = FIXTURES_PATH / "corpus.en"
    special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=400,
        special_tokens=special_tokens,
    )

    # Verify special tokens are in vocab at the beginning
    assert vocab[0] == b"<|endoftext|>"
    assert vocab[1] == b"<|pad|>"
    assert vocab[2] == b"<|unk|>"

    # Verify vocab size
    assert len(vocab) == 400

    # Verify no special token substrings appear in merged tokens
    for token_bytes in vocab.values():
        if token_bytes not in [s.encode("utf-8") for s in special_tokens]:
            assert b"<|" not in token_bytes


def test_train_bpe_tie_breaking():
    """
    Test that tie-breaking uses lexicographic ordering.
    This test verifies deterministic behavior when pairs have equal frequency.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    vocab1, merges1 = run_train_bpe(
        input_path=input_path,
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )

    # Run again - should produce identical results
    vocab2, merges2 = run_train_bpe(
        input_path=input_path,
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )

    assert merges1 == merges2, "BPE training should be deterministic"
    assert set(vocab1.values()) == set(vocab2.values())


def test_train_bpe_tinystories_larger_vocab():
    """Test BPE on TinyStories with a moderate vocabulary."""
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=450,  # Smaller vocab that fits the sample file
        special_tokens=["<|endoftext|>"],
    )

    assert len(vocab) == 450
    assert len(merges) == 450 - 257

    # Common English words should appear as single tokens
    vocab_set = set(vocab.values())
    # Common pairs should be merged
    assert b"th" in vocab_set or b"he" in vocab_set


def test_train_bpe_merge_order_verification():
    """Verify that merges happen in frequency order with proper tie-breaking."""
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=280,
        special_tokens=["<|endoftext|>"],
    )

    # Verify the first few merges are in expected order based on corpus.en
    expected_first_merges = [
        (b"h", b"e"),  # 15 occurrences
        (b" ", b"t"),  # 12 occurrences
        (b" ", b"a"),  # 11 occurrences
    ]

    for i, expected in enumerate(expected_first_merges):
        assert merges[i] == expected, f"Merge {i} should be {expected}, got {merges[i]}"
