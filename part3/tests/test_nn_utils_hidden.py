"""
Hidden tests for nn_utils - only available on Gradescope.
"""

import numpy
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from .adapters import (
    run_softmax,
    run_cross_entropy,
    run_gradient_clipping,
    run_token_accuracy,
    run_perplexity,
)


def test_softmax_high_dimensional():
    """Test softmax on higher-dimensional tensors."""
    torch.manual_seed(33333)
    x = torch.randn(4, 8, 16)
    expected = F.softmax(x, dim=-1)
    actual = run_softmax(x, dim=-1)
    numpy.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-6)


def test_softmax_extreme_values():
    """Test softmax with extreme values (numerical stability)."""
    x = torch.tensor([[-1000.0, 0.0, 1000.0]])
    expected = F.softmax(x, dim=-1)
    actual = run_softmax(x, dim=-1)
    numpy.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-6)


def test_cross_entropy_large_batch():
    torch.manual_seed(11111)
    logits = torch.randn(64, 100)
    targets = torch.randint(0, 100, (64,))
    expected = F.cross_entropy(logits, targets)
    actual = run_cross_entropy(logits, targets)
    numpy.testing.assert_allclose(actual.item(), expected.item(), atol=1e-4)


def test_cross_entropy_confident_correct():
    logits = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    targets = torch.tensor([0, 1])
    expected = F.cross_entropy(logits, targets)
    actual = run_cross_entropy(logits, targets)
    numpy.testing.assert_allclose(actual.item(), expected.item(), atol=1e-4)


def test_gradient_clipping_very_small_norm():
    torch.manual_seed(22222)
    tensors = [torch.randn((10, 10)) for _ in range(3)]
    max_norm = 1e-6
    t1 = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    loss = torch.cat(t1).sum()
    loss.backward()
    clip_grad_norm_(t1, max_norm)
    t1_grads = [torch.clone(t.grad) for t in t1]
    t2 = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    loss2 = torch.cat(t2).sum()
    loss2.backward()
    run_gradient_clipping(t2, max_norm)
    t2_grads = [torch.clone(t.grad) for t in t2]
    for g1, g2 in zip(t1_grads, t2_grads):
        numpy.testing.assert_allclose(g1.numpy(), g2.numpy(), atol=1e-10)


def test_token_accuracy_large_vocab():
    torch.manual_seed(44444)
    logits = torch.randn(100, 10000)
    targets = logits.argmax(dim=-1).clone()
    wrong_indices = torch.randperm(100)[:20]
    targets[wrong_indices] = (targets[wrong_indices] + 1) % 10000
    accuracy = run_token_accuracy(logits, targets)
    numpy.testing.assert_allclose(accuracy.item(), 0.80, atol=0.01)


def test_perplexity_large_vocab():
    torch.manual_seed(55555)
    logits = torch.randn(50, 5000)
    targets = torch.randint(0, 5000, (50,))
    ppl = run_perplexity(logits, targets)
    expected_ppl = torch.exp(F.cross_entropy(logits, targets))
    # Use relative tolerance for large values
    numpy.testing.assert_allclose(ppl.item(), expected_ppl.item(), rtol=1e-5, atol=0.1)


def test_perplexity_binary():
    logits = torch.zeros(4, 2)
    targets = torch.tensor([0, 1, 0, 1])
    ppl = run_perplexity(logits, targets)
    numpy.testing.assert_allclose(ppl.item(), 2.0, atol=1e-4)
