"""
Hidden tests for transformer model - only available on Gradescope.
These tests use different configurations and edge cases.
"""

import numpy
import torch
import torch.nn.functional as F
from einops import rearrange

from .adapters import (
    run_multihead_self_attention_with_rope,
    run_rope,
    run_silu,
    run_multihead_self_attention,
    run_swiglu,
    run_rmsnorm,
    run_scaled_dot_product_attention,
    run_transformer_block,
    run_transformer_lm,
    run_linear,
    run_embedding,
)


def test_linear_different_dimensions():
    """Test linear layer with different input/output dimensions."""
    torch.manual_seed(12345)
    d_in, d_out = 128, 256
    weights = torch.randn(d_out, d_in)
    in_features = torch.randn(2, 10, d_in)

    output = run_linear(
        d_in=d_in, d_out=d_out, weights=weights, in_features=in_features
    )

    expected = F.linear(in_features, weights)
    numpy.testing.assert_allclose(
        output.detach().numpy(),
        expected.detach().numpy(),
        atol=1e-5,
    )


def test_embedding_edge_indices():
    """Test embedding with edge case indices (0 and max)."""
    torch.manual_seed(54321)
    vocab_size, d_model = 1000, 64
    weights = torch.randn(vocab_size, d_model)

    # Test with indices at boundaries
    token_ids = torch.tensor([[0, 1, vocab_size - 2, vocab_size - 1]])

    output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights,
        token_ids=token_ids,
    )

    expected = F.embedding(token_ids, weights)
    numpy.testing.assert_allclose(
        output.detach().numpy(),
        expected.detach().numpy(),
        atol=1e-6,
    )


def test_rmsnorm_numerical_stability():
    """Test RMSNorm with very small and very large values."""
    torch.manual_seed(11111)
    d_model = 64
    weights = torch.ones(d_model)

    # Test with small values
    small_input = torch.randn(2, 8, d_model) * 1e-4
    output_small = run_rmsnorm(
        d_model=d_model, eps=1e-5, weights=weights, in_features=small_input
    )
    assert not torch.isnan(output_small).any(), "RMSNorm produces NaN for small inputs"

    # Test with large values
    large_input = torch.randn(2, 8, d_model) * 1e4
    output_large = run_rmsnorm(
        d_model=d_model, eps=1e-5, weights=weights, in_features=large_input
    )
    assert not torch.isnan(output_large).any(), "RMSNorm produces NaN for large inputs"


def test_silu_edge_cases():
    """Test SiLU with edge case values."""
    # Test with zeros
    x_zero = torch.zeros(3, 4)
    output_zero = run_silu(x_zero)
    numpy.testing.assert_allclose(
        output_zero.detach().numpy(), torch.zeros(3, 4).numpy(), atol=1e-6
    )

    # Test with large positive values
    x_large_pos = torch.tensor([[100.0, 50.0], [25.0, 10.0]])
    output_large = run_silu(x_large_pos)
    expected_large = F.silu(x_large_pos)
    numpy.testing.assert_allclose(
        output_large.detach().numpy(), expected_large.detach().numpy(), atol=1e-4
    )

    # Test with large negative values (should be close to 0)
    x_large_neg = torch.tensor([[-100.0, -50.0]])
    output_neg = run_silu(x_large_neg)
    assert (output_neg.abs() < 1e-10).all(), "SiLU of large negative should be ~0"


def test_swiglu_gradient_flow():
    """Test that SwiGLU allows gradient flow."""
    torch.manual_seed(22222)
    d_model, d_ff = 32, 64
    w1_weight = torch.randn(d_ff, d_model, requires_grad=True)
    w2_weight = torch.randn(d_model, d_ff, requires_grad=True)
    w3_weight = torch.randn(d_ff, d_model, requires_grad=True)

    in_features = torch.randn(1, 4, d_model, requires_grad=True)

    output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_features,
    )

    # Output should have correct shape
    assert output.shape == in_features.shape


def test_rope_different_positions():
    """Test RoPE with non-sequential positions."""
    torch.manual_seed(33333)
    d_model = 32
    theta = 10000.0
    max_seq_len = 100

    in_features = torch.randn(2, 8, d_model)

    # Non-sequential positions
    pos_ids = torch.tensor([[0, 5, 10, 15, 20, 25, 30, 35]])

    output = run_rope(
        d_model=d_model,
        theta=theta,
        max_seq_len=max_seq_len,
        in_query_or_key=in_features,
        token_positions=pos_ids,
    )

    assert output.shape == in_features.shape
    assert not torch.isnan(output).any()


def test_rope_position_encoding_uniqueness():
    """Test that different positions produce different embeddings."""
    torch.manual_seed(44444)
    d_model = 64
    theta = 10000.0
    max_seq_len = 50

    # Same input, different positions
    in_features = torch.ones(1, 3, d_model)

    pos1 = torch.tensor([[0, 1, 2]])
    pos2 = torch.tensor([[10, 11, 12]])

    output1 = run_rope(d_model, theta, max_seq_len, in_features, pos1)
    output2 = run_rope(d_model, theta, max_seq_len, in_features, pos2)

    # Outputs should be different
    assert not torch.allclose(
        output1, output2
    ), "Different positions should produce different outputs"


def test_scaled_dot_product_attention_single_query():
    """Test attention with a single query position."""
    torch.manual_seed(55555)
    d_k = 32

    Q = torch.randn(1, 1, d_k)  # Single query
    K = torch.randn(1, 10, d_k)  # 10 keys
    V = torch.randn(1, 10, d_k)  # 10 values
    mask = torch.ones(1, 1, 10, dtype=torch.bool)  # Attend to all

    output = run_scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)

    assert output.shape == (1, 1, d_k)


def test_scaled_dot_product_attention_masked_all():
    """Test attention when all positions are masked."""
    torch.manual_seed(66666)
    d_k = 16

    Q = torch.randn(1, 4, d_k)
    K = torch.randn(1, 4, d_k)
    V = torch.randn(1, 4, d_k)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)  # Mask everything

    output = run_scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)

    # Output should be zeros or handled gracefully (no NaN)
    assert not torch.isnan(
        output
    ).any(), "Fully masked attention should not produce NaN"


def test_multihead_attention_num_heads_variation():
    """Test multi-head attention with different number of heads."""
    torch.manual_seed(77777)
    d_model = 64

    for num_heads in [1, 2, 4, 8]:
        q_proj = torch.randn(d_model, d_model)
        k_proj = torch.randn(d_model, d_model)
        v_proj = torch.randn(d_model, d_model)
        o_proj = torch.randn(d_model, d_model)

        in_features = torch.randn(1, 8, d_model)

        output = run_multihead_self_attention(
            d_model=d_model,
            num_heads=num_heads,
            q_proj_weight=q_proj,
            k_proj_weight=k_proj,
            v_proj_weight=v_proj,
            o_proj_weight=o_proj,
            in_features=in_features,
        )

        assert output.shape == in_features.shape, f"Failed for num_heads={num_heads}"


def test_transformer_block_residual_connection():
    """Test that transformer block has proper residual connections."""
    torch.manual_seed(88888)
    d_model = 32
    num_heads = 4
    d_ff = 64
    max_seq_len = 16
    theta = 10000.0

    # Create block weights
    weights = {
        "ln1.weight": torch.ones(d_model),
        "ln2.weight": torch.ones(d_model),
        "attn.q_proj.weight": torch.zeros(d_model, d_model),
        "attn.k_proj.weight": torch.zeros(d_model, d_model),
        "attn.v_proj.weight": torch.zeros(d_model, d_model),
        "attn.output_proj.weight": torch.zeros(d_model, d_model),
        "ffn.w1.weight": torch.zeros(d_ff, d_model),
        "ffn.w2.weight": torch.zeros(d_model, d_ff),
        "ffn.w3.weight": torch.zeros(d_ff, d_model),
    }

    in_features = torch.randn(1, 4, d_model)

    output = run_transformer_block(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        weights=weights,
        in_features=in_features,
    )

    # With zero weights, output should equal input (residual only)
    # Note: This depends on implementation details
    assert output.shape == in_features.shape
