"""
Hidden tests for tokenizer - only available on Gradescope.
These tests use different strings and edge cases to prevent reverse engineering.
"""

import json
import os

import tiktoken

from common import FIXTURES_PATH, gpt2_bytes_to_unicode
from tokenizer import get_tokenizer

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token
    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


def test_encode_decode_programming_code():
    """Test encoding/decoding of programming code with special characters."""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    test_string = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
"""

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert tokenizer.decode(ids) == test_string


def test_encode_decode_json_content():
    """Test encoding/decoding of JSON-like content."""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    test_string = (
        '{"name": "John", "age": 30, "city": "New York", "scores": [95, 87, 92]}'
    )

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert tokenizer.decode(ids) == test_string


def test_encode_decode_mixed_languages():
    """Test encoding/decoding of text with mixed languages."""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    test_string = "Hello world! Bonjour le monde! Hola mundo! 你好世界! مرحبا بالعالم"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert tokenizer.decode(ids) == test_string


def test_encode_decode_repeated_patterns():
    """Test encoding/decoding of repeated patterns."""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    test_string = "ab" * 100 + " " + "xyz" * 50

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert tokenizer.decode(ids) == test_string


def test_encode_decode_numbers_and_math():
    """Test encoding/decoding of mathematical expressions."""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    test_string = (
        "The equation is: E = mc² where c = 299,792,458 m/s and m = 1.67 × 10⁻²⁷ kg"
    )

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert tokenizer.decode(ids) == test_string


def test_encode_decode_whitespace_variations():
    """Test encoding/decoding with various whitespace patterns."""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    test_string = "word1  word2   word3\t\ttabbed\n\n\nnewlines    spaces"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert tokenizer.decode(ids) == test_string


def test_special_tokens_mid_word():
    """Test that special tokens are recognized even without whitespace."""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )

    test_string = "word<|endoftext|>another"
    ids = tokenizer.encode(test_string)
    decoded = tokenizer.decode(ids)
    assert decoded == test_string

    # Verify the special token is its own token
    tokenized = [tokenizer.decode([x]) for x in ids]
    assert "<|endoftext|>" in tokenized


def test_encode_iterable_consistency():
    """Test that encode_iterable produces same results as encode."""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )

    test_string = "This is a test string that spans multiple lines.\nIt has various content.\nAnd more lines here."

    # Encode the full string
    full_ids = tokenizer.encode(test_string)

    # Encode via iterable (preserving newlines using splitlines with keepends=True)
    iterable_ids = list(
        tokenizer.encode_iterable(test_string.splitlines(keepends=True))
    )

    # Results should match when decoded
    assert tokenizer.decode(full_ids) == tokenizer.decode(iterable_ids)
