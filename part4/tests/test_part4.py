"""
Tests for Part 4 components.
"""

import pytest
import torch
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from part1.tokenizer import get_tokenizer
from part1.train_bpe import train_bpe
from part2.model import TransformerLM
from part4.sampling import greedy_decode, top_k_decode
from part4.datasets import PretrainingDataset, MultipleChoiceQADataset
from part4.qa_model import TransformerForMultipleChoice
from part4.prompting import PromptTemplate, PromptingPipeline


@pytest.fixture
def tiny_tokenizer():
    """Create a small tokenizer for testing."""
    # Create a minimal vocabulary for testing
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    merges = []
    return get_tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])


@pytest.fixture
def tiny_model(tiny_tokenizer):
    """Create a small transformer model for testing."""
    return TransformerLM(
        vocab_size=257,  # 256 bytes + 1 special token
        context_length=64,
        d_model=32,
        num_layers=2,
        num_heads=2,
        d_ff=64,
    )


class TestSampling:
    """Test sampling functions."""

    def test_greedy_decode_shape(self, tiny_model, tiny_tokenizer):
        """Test that greedy decode returns correct shape."""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        output = greedy_decode(tiny_model, input_ids, max_new_tokens=10)

        assert output.shape[0] == 1
        assert output.shape[1] == 4 + 10  # input + generated

    def test_top_k_decode_shape(self, tiny_model, tiny_tokenizer):
        """Test that top-k decode returns correct shape."""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        output = top_k_decode(tiny_model, input_ids, max_new_tokens=10, k=10)

        assert output.shape[0] == 1
        assert output.shape[1] == 4 + 10


class TestDatasets:
    """Test dataset classes."""

    def test_pretraining_dataset(self, tiny_tokenizer, tmp_path):
        """Test PretrainingDataset."""
        # Create temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Once upon a time there was a little cat." * 10)

        dataset = PretrainingDataset(
            file_path=test_file,
            tokenizer=tiny_tokenizer,
            max_length=32,
            stride=16,
        )

        assert len(dataset) > 0

        sample = dataset[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (32,)
        assert sample["labels"].shape == (32,)

    def test_qa_dataset(self, tiny_tokenizer):
        """Test MultipleChoiceQADataset."""
        data = [
            {
                "context": "The cat sat on the mat.",
                "question": "Where did the cat sit?",
                "choices": ["On the mat", "On the chair", "On the bed", "On the floor"],
                "answer": 0,
            }
        ]

        dataset = MultipleChoiceQADataset(
            data=data,
            tokenizer=tiny_tokenizer,
            max_length=32,
            num_choices=4,
        )

        assert len(dataset) == 1

        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (4, 32)
        assert sample["attention_mask"].shape == (4, 32)
        assert sample["labels"].item() == 0


class TestQAModel:
    """Test QA model."""

    def test_qa_model_forward(self, tiny_model):
        """Test QA model forward pass."""
        qa_model = TransformerForMultipleChoice(
            transformer_lm=tiny_model,
            hidden_size=32,
            num_choices=4,
            pooling="last",
        )

        # Batch of 2 examples, 4 choices each, 16 tokens
        input_ids = torch.randint(0, 256, (2, 4, 16))
        attention_mask = torch.ones(2, 4, 16)

        logits = qa_model(input_ids, attention_mask)

        assert logits.shape == (2, 4)

    def test_qa_model_predict(self, tiny_model):
        """Test QA model prediction."""
        qa_model = TransformerForMultipleChoice(
            transformer_lm=tiny_model,
            hidden_size=32,
            num_choices=4,
            pooling="last",
        )

        input_ids = torch.randint(0, 256, (2, 4, 16))
        attention_mask = torch.ones(2, 4, 16)

        predictions = qa_model.predict(input_ids, attention_mask)

        assert predictions.shape == (2,)
        assert (predictions >= 0).all() and (predictions < 4).all()


class TestPrompting:
    """Test prompting utilities."""

    def test_prompt_template_basic(self):
        """Test basic prompt template."""
        template = PromptTemplate(template_name="basic")

        prompt = template.format(
            context="The sky is blue.",
            question="What color is the sky?",
            choices=["Red", "Blue", "Green", "Yellow"],
        )

        assert "The sky is blue" in prompt
        assert "What color is the sky" in prompt
        assert "A. Red" in prompt
        assert "B. Blue" in prompt

    def test_prompt_template_custom(self):
        """Test custom prompt template."""
        custom = "{context}\n{question}\n{choices_formatted}"
        template = PromptTemplate(custom_template=custom)

        prompt = template.format(
            context="Hello",
            question="World?",
            choices=["A", "B"],
        )

        assert "Hello" in prompt
        assert "World?" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
