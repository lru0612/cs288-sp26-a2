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

from model import (
    Linear,
    Embedding,
    RMSNorm,
    silu,
    softmax,
    SwiGLU,
    RotaryPositionEmbedding,
    apply_rope,
    scaled_dot_product_attention,
    MultiHeadSelfAttention,
    MultiHeadSelfAttentionWithRoPE,
    TransformerBlock,
    TransformerLM,
    count_parameters,
    count_flops_per_token,
    estimate_memory_bytes,
)


def run_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Run softmax function."""
    return softmax(x, dim=dim)


def run_silu(x: Tensor) -> Tensor:
    """Run SiLU activation."""
    return silu(x)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    """
    Run linear transformation with provided weights.

    Args:
        d_in: Input dimension
        d_out: Output dimension
        weights: Weight matrix of shape (d_out, d_in)
        in_features: Input tensor of shape (..., d_in)

    Returns:
        Output tensor of shape (..., d_out)
    """
    linear = Linear(d_in, d_out)
    linear.weight.data.copy_(weights)
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Tensor,
    token_ids: Tensor,
) -> Tensor:
    """
    Run embedding lookup with provided weights.

    Args:
        vocab_size: Vocabulary size
        d_model: Embedding dimension
        weights: Embedding weights of shape (vocab_size, d_model)
        token_ids: Token indices

    Returns:
        Embedded tokens
    """
    embedding = Embedding(vocab_size, d_model)
    embedding.weight.data.copy_(weights)
    return embedding(token_ids)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    """
    Run RMSNorm with provided weights.

    Args:
        d_model: Model dimension
        eps: Epsilon for numerical stability
        weights: Scale weights of shape (d_model,)
        in_features: Input tensor

    Returns:
        Normalized tensor
    """
    rmsnorm = RMSNorm(d_model, eps)
    rmsnorm.weight.data.copy_(weights)
    return rmsnorm(in_features)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """
    Run SwiGLU feedforward with provided weights.

    Args:
        d_model: Model dimension
        d_ff: Feedforward hidden dimension
        w1_weight: Gate projection weights (d_ff, d_model)
        w2_weight: Down projection weights (d_model, d_ff)
        w3_weight: Up projection weights (d_ff, d_model)
        in_features: Input tensor

    Returns:
        Output tensor
    """
    swiglu = SwiGLU(d_model, d_ff)
    swiglu.w1.weight.data.copy_(w1_weight)
    swiglu.w2.weight.data.copy_(w2_weight)
    swiglu.w3.weight.data.copy_(w3_weight)
    return swiglu(in_features)


def run_rope(
    d_model: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
) -> Tensor:
    """
    Run RoPE on query or key tensor.

    Args:
        d_model: Head dimension
        theta: RoPE base frequency
        max_seq_len: Maximum sequence length
        in_query_or_key: Input tensor
        token_positions: Position indices

    Returns:
        Tensor with RoPE applied
    """
    rope = RotaryPositionEmbedding(d_model, max_seq_len, theta)
    return rope(in_query_or_key, token_positions)


def run_scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Run scaled dot-product attention.

    Args:
        Q: Query tensor
        K: Key tensor
        V: Value tensor
        mask: Attention mask

    Returns:
        Attention output
    """
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """
    Run multi-head self-attention with provided weights.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        q_proj_weight: Query projection weights
        k_proj_weight: Key projection weights
        v_proj_weight: Value projection weights
        o_proj_weight: Output projection weights
        in_features: Input tensor

    Returns:
        Attention output
    """
    attn = MultiHeadSelfAttention(d_model, num_heads)
    attn.q_proj.weight.data.copy_(q_proj_weight)
    attn.k_proj.weight.data.copy_(k_proj_weight)
    attn.v_proj.weight.data.copy_(v_proj_weight)
    attn.output_proj.weight.data.copy_(o_proj_weight)
    return attn(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor,
) -> Tensor:
    """
    Run multi-head self-attention with RoPE.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        theta: RoPE base frequency
        q_proj_weight: Query projection weights
        k_proj_weight: Key projection weights
        v_proj_weight: Value projection weights
        o_proj_weight: Output projection weights
        in_features: Input tensor
        token_positions: Position indices

    Returns:
        Attention output
    """
    attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
    attn.q_proj.weight.data.copy_(q_proj_weight)
    attn.k_proj.weight.data.copy_(k_proj_weight)
    attn.v_proj.weight.data.copy_(v_proj_weight)
    attn.output_proj.weight.data.copy_(o_proj_weight)
    return attn(in_features, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict,
    in_features: Tensor,
) -> Tensor:
    """
    Run a Transformer block with provided weights.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feedforward hidden dimension
        max_seq_len: Maximum sequence length
        theta: RoPE base frequency
        weights: Dictionary of weights
        in_features: Input tensor

    Returns:
        Transformer block output
    """
    block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)

    # Load weights
    if "ln1.weight" in weights:
        block.ln1.weight.data.copy_(weights["ln1.weight"])
    if "ln2.weight" in weights:
        block.ln2.weight.data.copy_(weights["ln2.weight"])

    if "attn.q_proj.weight" in weights:
        block.attn.q_proj.weight.data.copy_(weights["attn.q_proj.weight"])
    if "attn.k_proj.weight" in weights:
        block.attn.k_proj.weight.data.copy_(weights["attn.k_proj.weight"])
    if "attn.v_proj.weight" in weights:
        block.attn.v_proj.weight.data.copy_(weights["attn.v_proj.weight"])
    if "attn.output_proj.weight" in weights:
        block.attn.output_proj.weight.data.copy_(weights["attn.output_proj.weight"])

    if "ffn.w1.weight" in weights:
        block.ffn.w1.weight.data.copy_(weights["ffn.w1.weight"])
    if "ffn.w2.weight" in weights:
        block.ffn.w2.weight.data.copy_(weights["ffn.w2.weight"])
    if "ffn.w3.weight" in weights:
        block.ffn.w3.weight.data.copy_(weights["ffn.w3.weight"])

    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict,
    in_indices: Tensor,
) -> Tensor:
    """
    Run a Transformer LM with provided weights.

    Args:
        vocab_size: Vocabulary size
        context_length: Maximum context length
        d_model: Model dimension
        num_layers: Number of Transformer blocks
        num_heads: Number of attention heads
        d_ff: Feedforward hidden dimension
        rope_theta: RoPE base frequency
        weights: Dictionary of weights
        in_indices: Input token indices

    Returns:
        Logits tensor
    """
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    model.load_weights(weights)
    return model(in_indices)


# =============================================================================
# Transformer Accounting Functions
# =============================================================================


def run_count_parameters(model) -> int:
    """Count the total number of parameters in a model."""
    return count_parameters(model)


def run_count_flops_per_token(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
) -> int:
    """Estimate FLOPs per token for forward pass."""
    return count_flops_per_token(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )


def run_estimate_memory_bytes(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
    dtype_bytes: int = 4,
) -> int:
    """Estimate memory required for model parameters."""
    return estimate_memory_bytes(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        d_ff=d_ff,
        dtype_bytes=dtype_bytes,
    )


def create_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float = 10000.0,
) -> TransformerLM:
    """Create a TransformerLM model without loading weights."""
    return TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
