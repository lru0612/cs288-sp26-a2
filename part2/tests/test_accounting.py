"""
Tests for Transformer LM resource accounting functions.
"""

import pytest

from .adapters import (
    run_count_parameters,
    run_count_flops_per_token,
    run_estimate_memory_bytes,
    create_transformer_lm,
)


def test_count_parameters(vocab_size, d_model, n_layers, n_heads, d_ff, n_keys):
    """Test that parameter counting works correctly."""
    model = create_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
    )

    param_count = run_count_parameters(model)

    # Manually calculate expected parameters:
    # Token embeddings: vocab_size * d_model
    # Output projection: vocab_size * d_model (not tied)
    # Final layer norm: d_model
    # Per layer:
    #   - ln1, ln2: 2 * d_model
    #   - attention: 4 * d_model * d_model (Q, K, V, O projections)
    #   - ffn: 3 * d_model * d_ff (w1, w2, w3 for SwiGLU)

    expected = (
        vocab_size * d_model  # token_embeddings
        + vocab_size * d_model  # output
        + d_model  # final_ln
        + n_layers
        * (
            2 * d_model  # ln1, ln2
            + 4 * d_model * d_model  # attention projections
            + 3
            * d_model
            * d_ff  # ffn (w1: d_ff*d_model, w2: d_model*d_ff, w3: d_ff*d_model)
        )
    )

    assert param_count == expected, f"Expected {expected} parameters, got {param_count}"


def test_count_parameters_scales_with_layers():
    """Test that parameter count scales linearly with number of layers."""
    vocab_size = 256
    d_model = 64
    d_ff = 256
    n_heads = 4
    n_keys = 32

    model_1layer = create_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=1,
        num_heads=n_heads,
        d_ff=d_ff,
    )

    model_2layer = create_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=2,
        num_heads=n_heads,
        d_ff=d_ff,
    )

    params_1 = run_count_parameters(model_1layer)
    params_2 = run_count_parameters(model_2layer)

    # Per-layer params
    per_layer_params = params_2 - params_1

    # Check that adding a layer adds the expected number of parameters
    expected_per_layer = (
        2 * d_model  # ln1, ln2
        + 4 * d_model * d_model  # attention projections
        + 3 * d_model * d_ff  # ffn
    )

    assert per_layer_params == expected_per_layer


def test_flops_per_token_is_positive():
    """Test that FLOPs estimate is positive and reasonable."""
    vocab_size = 256
    context_length = 32
    d_model = 64
    num_layers = 2
    num_heads = 4
    d_ff = 256

    flops = run_count_flops_per_token(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    assert flops > 0, "FLOPs should be positive"

    # FLOPs should be at least the cost of the output projection
    min_flops = 2 * d_model * vocab_size
    assert (
        flops >= min_flops
    ), f"FLOPs should be at least {min_flops} (output projection cost)"


def test_flops_scales_with_layers():
    """Test that FLOPs scale approximately linearly with layers."""
    vocab_size = 256
    context_length = 32
    d_model = 64
    num_heads = 4
    d_ff = 256

    flops_1 = run_count_flops_per_token(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=1,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    flops_2 = run_count_flops_per_token(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=2,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    # The difference should be roughly equal (the per-layer cost)
    per_layer_flops = flops_2 - flops_1

    # Per-layer flops should be positive
    assert per_layer_flops > 0, "Per-layer FLOPs should be positive"


def test_memory_estimate_is_positive():
    """Test that memory estimate is positive and reasonable."""
    vocab_size = 256
    d_model = 64
    num_layers = 2
    d_ff = 256

    memory_bytes = run_estimate_memory_bytes(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        d_ff=d_ff,
        dtype_bytes=4,  # float32
    )

    assert memory_bytes > 0, "Memory estimate should be positive"

    # Memory should be at least the embedding table size
    min_memory = vocab_size * d_model * 4  # token embeddings in float32
    assert memory_bytes >= min_memory


def test_memory_scales_with_dtype():
    """Test that memory estimate scales with dtype size."""
    vocab_size = 256
    d_model = 64
    num_layers = 2
    d_ff = 256

    memory_fp32 = run_estimate_memory_bytes(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        d_ff=d_ff,
        dtype_bytes=4,  # float32
    )

    memory_fp16 = run_estimate_memory_bytes(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        d_ff=d_ff,
        dtype_bytes=2,  # float16
    )

    # fp32 memory should be 2x fp16 memory
    assert memory_fp32 == 2 * memory_fp16
