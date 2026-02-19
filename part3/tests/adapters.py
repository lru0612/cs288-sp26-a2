"""
Adapters for testing - provides interface between tests and implementation.
"""

import torch
from torch import Tensor

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from nn_utils import (
    softmax,
    cross_entropy,
    gradient_clipping,
    token_accuracy,
    perplexity,
)


def run_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Run softmax implementation."""
    return softmax(x, dim=dim)


def run_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Run cross-entropy implementation."""
    return cross_entropy(logits, targets)


def run_gradient_clipping(parameters, max_norm: float) -> Tensor:
    """Run gradient clipping implementation."""
    return gradient_clipping(parameters, max_norm)


def run_token_accuracy(
    logits: Tensor, targets: Tensor, ignore_index: int = -100
) -> Tensor:
    """Run token accuracy implementation."""
    return token_accuracy(logits, targets, ignore_index)


def run_perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """Run perplexity implementation."""
    return perplexity(logits, targets, ignore_index)
