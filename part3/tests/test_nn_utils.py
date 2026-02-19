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


def test_softmax_matches_pytorch():
    """Test that our softmax implementation matches PyTorch's."""
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )

    expected_output = F.softmax(x, dim=-1)
    actual_output = run_softmax(x, dim=-1)

    numpy.testing.assert_allclose(
        actual_output.detach().numpy(),
        expected_output.detach().numpy(),
        atol=1e-6,
    )

    # Test that softmax handles numerical overflow issues
    numpy.testing.assert_allclose(
        run_softmax(x + 100, dim=-1).detach().numpy(),
        expected_output.detach().numpy(),
        atol=1e-6,
    )


def test_cross_entropy():
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    numpy.testing.assert_allclose(
        run_cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
        .detach()
        .numpy(),
        expected.detach().numpy(),
        atol=1e-4,
    )

    # Test that cross-entropy handles numerical overflow issues
    large_inputs = 1000.0 * inputs
    large_expected_cross_entropy = F.cross_entropy(
        large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1)
    )
    numpy.testing.assert_allclose(
        run_cross_entropy(
            large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1)
        )
        .detach()
        .numpy(),
        large_expected_cross_entropy.detach().numpy(),
        atol=1e-4,
    )


def test_gradient_clipping():
    tensors = [torch.randn((5, 5)) for _ in range(6)]
    max_norm = 1e-2

    t1 = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    # Test freezing one parameter.
    t1[-1].requires_grad_(False)

    loss = torch.cat(t1).sum()
    loss.backward()
    clip_grad_norm_(t1, max_norm)
    t1_grads = [torch.clone(t.grad) for t in t1 if t.grad is not None]

    t1_c = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    t1_c[-1].requires_grad_(False)
    loss_c = torch.cat(t1_c).sum()
    loss_c.backward()
    run_gradient_clipping(t1_c, max_norm)
    t1_c_grads = [torch.clone(t.grad) for t in t1_c if t.grad is not None]

    assert len(t1_grads) == len(t1_c_grads)

    for t1_grad, t1_c_grad in zip(t1_grads, t1_c_grads):
        numpy.testing.assert_allclose(
            t1_grad.detach().numpy(),
            t1_c_grad.detach().numpy(),
            atol=1e-6,
        )


def test_token_accuracy_all_correct():
    """Test token accuracy when all predictions are correct."""
    # Logits where argmax matches targets exactly
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.5],  # argmax = 0
            [0.1, 3.0, 0.2],  # argmax = 1
            [1.0, 0.5, 2.5],  # argmax = 2
        ]
    )
    targets = torch.tensor([0, 1, 2])

    accuracy = run_token_accuracy(logits, targets)
    numpy.testing.assert_allclose(accuracy.item(), 1.0, atol=1e-6)


def test_token_accuracy_partial_correct():
    """Test token accuracy with some incorrect predictions."""
    logits = torch.tensor(
        [
            [2.0, 1.0],  # argmax = 0, target = 1 (wrong)
            [0.1, 3.0],  # argmax = 1, target = 1 (correct)
            [1.0, 0.5],  # argmax = 0, target = 0 (correct)
        ]
    )
    targets = torch.tensor([1, 1, 0])

    accuracy = run_token_accuracy(logits, targets)
    # 2 out of 3 correct = 0.6667
    numpy.testing.assert_allclose(accuracy.item(), 2.0 / 3.0, atol=1e-4)


def test_token_accuracy_with_ignore_index():
    """Test token accuracy ignores positions with ignore_index."""
    logits = torch.tensor(
        [
            [2.0, 1.0],  # argmax = 0, target = 0 (correct)
            [0.1, 3.0],  # argmax = 1, target = -100 (ignored)
            [1.0, 0.5],  # argmax = 0, target = 1 (wrong)
            [0.5, 2.0],  # argmax = 1, target = 1 (correct)
        ]
    )
    targets = torch.tensor([0, -100, 1, 1])

    accuracy = run_token_accuracy(logits, targets, ignore_index=-100)
    # Only 3 valid tokens, 2 correct = 0.6667
    numpy.testing.assert_allclose(accuracy.item(), 2.0 / 3.0, atol=1e-4)


def test_token_accuracy_all_wrong():
    """Test token accuracy when all predictions are wrong."""
    logits = torch.tensor(
        [
            [2.0, 1.0],  # argmax = 0
            [0.1, 3.0],  # argmax = 1
        ]
    )
    targets = torch.tensor([1, 0])  # Opposite of predictions

    accuracy = run_token_accuracy(logits, targets)
    numpy.testing.assert_allclose(accuracy.item(), 0.0, atol=1e-6)


def test_perplexity_near_perfect():
    """Test perplexity with near-perfect predictions (should be close to 1)."""
    # High confidence correct predictions
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],  # Confident prediction of class 0
            [0.0, 10.0, 0.0],  # Confident prediction of class 1
            [0.0, 0.0, 10.0],  # Confident prediction of class 2
        ]
    )
    targets = torch.tensor([0, 1, 2])

    ppl = run_perplexity(logits, targets)
    # Should be very close to 1.0 (perfect prediction)
    assert ppl.item() < 1.1, f"Expected perplexity close to 1, got {ppl.item()}"


def test_perplexity_uniform():
    """Test perplexity with uniform predictions (maximum uncertainty)."""
    # Uniform distribution over 3 classes
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 1, 2])

    ppl = run_perplexity(logits, targets)
    # Uniform over 3 classes should give perplexity = 3
    numpy.testing.assert_allclose(ppl.item(), 3.0, atol=1e-4)


def test_perplexity_with_ignore_index():
    """Test perplexity ignores positions with ignore_index."""
    # Mix of confident and ignored positions
    logits = torch.tensor(
        [
            [10.0, 0.0],  # Confident correct
            [0.0, 0.0],  # Uniform (will be ignored)
            [10.0, 0.0],  # Confident correct
        ]
    )
    targets = torch.tensor([0, -100, 0])

    ppl = run_perplexity(logits, targets, ignore_index=-100)
    # Only considers the confident predictions, should be close to 1
    assert ppl.item() < 1.1, f"Expected perplexity close to 1, got {ppl.item()}"


def test_perplexity_matches_exp_cross_entropy():
    """Test that perplexity equals exp(cross_entropy)."""
    logits = torch.tensor(
        [
            [1.5, 0.8, 0.3],
            [0.2, 2.1, 0.9],
            [0.7, 0.4, 1.8],
            [1.2, 1.1, 0.6],
        ]
    )
    targets = torch.tensor([0, 1, 2, 0])

    ppl = run_perplexity(logits, targets)
    ce_loss = F.cross_entropy(logits, targets)
    expected_ppl = torch.exp(ce_loss)

    numpy.testing.assert_allclose(ppl.item(), expected_ppl.item(), atol=1e-4)
