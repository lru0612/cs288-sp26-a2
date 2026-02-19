"""
Pytest configuration and fixtures for Part 2 tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add parent directory to path so we can import model and nn_utils
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class NumpySnapshot:
    """
    Simple snapshot testing utility for numpy arrays.
    Stores expected values and compares against actual outputs.
    """

    def __init__(self, request, snapshots_dir):
        self.test_name = request.node.name
        self.snapshots_dir = snapshots_dir
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _get_snapshot_path(self):
        return self.snapshots_dir / f"{self.test_name}.npy"

    def assert_match(self, actual, atol=1e-6, rtol=1e-5):
        """Assert that actual matches the stored snapshot."""
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().cpu().numpy()

        snapshot_path = self._get_snapshot_path()

        if not snapshot_path.exists():
            # Create new snapshot
            np.save(snapshot_path, actual)
            pytest.skip(f"Created new snapshot at {snapshot_path}")

        expected = np.load(snapshot_path)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.fixture
def numpy_snapshot(request):
    """Fixture for snapshot testing numpy arrays."""
    snapshots_dir = Path(__file__).parent / "__snapshots__"
    return NumpySnapshot(request, snapshots_dir)


# Model architecture parameters
@pytest.fixture
def d_model():
    """Model embedding dimension."""
    return 64


@pytest.fixture
def d_ff(d_model):
    """Feed-forward hidden dimension (typically 4x d_model or 8/3 * d_model for SwiGLU)."""
    return d_model * 4


@pytest.fixture
def n_heads():
    """Number of attention heads."""
    return 4


@pytest.fixture
def n_layers():
    """Number of transformer layers."""
    return 2


@pytest.fixture
def vocab_size():
    """Vocabulary size."""
    return 256


@pytest.fixture
def n_keys():
    """Maximum sequence length / context length."""
    return 32


@pytest.fixture
def n_queries(n_keys):
    """Query sequence length (same as key length for self-attention)."""
    return n_keys


@pytest.fixture
def theta():
    """RoPE base frequency."""
    return 10000.0


@pytest.fixture
def batch_size():
    """Batch size for test inputs."""
    return 2


@pytest.fixture
def seq_len(n_keys):
    """Sequence length for test inputs."""
    return n_keys


# Input tensors
@pytest.fixture
def in_embeddings(batch_size, seq_len, d_model):
    """Input embeddings tensor of shape (batch, seq_len, d_model)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, d_model)


@pytest.fixture
def in_indices(batch_size, seq_len, vocab_size):
    """Input token indices of shape (batch, seq_len)."""
    torch.manual_seed(42)
    return torch.randint(0, vocab_size, (batch_size, seq_len))


@pytest.fixture
def pos_ids(seq_len):
    """Position IDs tensor."""
    return torch.arange(seq_len)


# Attention test tensors
@pytest.fixture
def d_k(d_model, n_heads):
    """Key/query dimension per head."""
    return d_model // n_heads


@pytest.fixture
def q(batch_size, n_heads, seq_len, d_k):
    """Query tensor for attention tests. Shape: (batch*heads, seq_len, d_k)."""
    torch.manual_seed(42)
    return torch.randn(batch_size * n_heads, seq_len, d_k)


@pytest.fixture
def k(batch_size, n_heads, seq_len, d_k):
    """Key tensor for attention tests. Shape: (batch*heads, seq_len, d_k)."""
    torch.manual_seed(43)
    return torch.randn(batch_size * n_heads, seq_len, d_k)


@pytest.fixture
def v(batch_size, n_heads, seq_len, d_k):
    """Value tensor for attention tests. Shape: (batch*heads, seq_len, d_k)."""
    torch.manual_seed(44)
    return torch.randn(batch_size * n_heads, seq_len, d_k)


@pytest.fixture
def mask(batch_size, n_heads, seq_len):
    """Causal attention mask. Shape: (batch*heads, seq_len, seq_len)."""
    # Create causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    # Expand for batch and heads
    mask = causal_mask.unsqueeze(0).expand(batch_size * n_heads, -1, -1)
    return mask


# State dict with pretrained weights for testing
@pytest.fixture
def ts_state_dict(d_model, d_ff, n_heads, n_layers, vocab_size):
    """
    Create a mock transformer state dict with deterministic weights.
    Returns tuple of (state_dict, config).
    """
    torch.manual_seed(42)

    state_dict = {}

    # Token embeddings
    state_dict["token_embeddings.weight"] = torch.randn(vocab_size, d_model) * 0.02

    # Output layer (tied with embeddings or separate)
    state_dict["output.weight"] = torch.randn(vocab_size, d_model) * 0.02

    # Final layer norm
    state_dict["final_ln.weight"] = torch.ones(d_model)

    # Per-layer weights
    for layer_idx in range(n_layers):
        prefix = f"layers.{layer_idx}"

        # Attention layer norms
        state_dict[f"{prefix}.ln1.weight"] = torch.ones(d_model)
        state_dict[f"{prefix}.ln2.weight"] = torch.ones(d_model)

        # Attention projections
        state_dict[f"{prefix}.attn.q_proj.weight"] = (
            torch.randn(d_model, d_model) * 0.02
        )
        state_dict[f"{prefix}.attn.k_proj.weight"] = (
            torch.randn(d_model, d_model) * 0.02
        )
        state_dict[f"{prefix}.attn.v_proj.weight"] = (
            torch.randn(d_model, d_model) * 0.02
        )
        state_dict[f"{prefix}.attn.output_proj.weight"] = (
            torch.randn(d_model, d_model) * 0.02
        )

        # FFN weights (SwiGLU uses w1, w2, w3)
        state_dict[f"{prefix}.ffn.w1.weight"] = torch.randn(d_ff, d_model) * 0.02
        state_dict[f"{prefix}.ffn.w2.weight"] = torch.randn(d_model, d_ff) * 0.02
        state_dict[f"{prefix}.ffn.w3.weight"] = torch.randn(d_ff, d_model) * 0.02

    config = {
        "d_model": d_model,
        "d_ff": d_ff,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "vocab_size": vocab_size,
    }

    return state_dict, config
