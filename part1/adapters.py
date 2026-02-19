"""
Adapters for testing - provides interface between tests and implementation.
"""

from pathlib import Path

from train_bpe import train_bpe


def run_train_bpe(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer from a text file.

    Args:
        input_path: Path to the input text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include

    Returns:
        Tuple of (vocab, merges) where:
        - vocab: dict mapping token_id (int) -> token (bytes)
        - merges: list of merge pairs (bytes, bytes)
    """
    return train_bpe(input_path, vocab_size, special_tokens)
