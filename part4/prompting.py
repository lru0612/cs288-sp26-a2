"""
Prompting utilities for multiple-choice QA.
Example submission.
"""

import random
import torch
from torch import Tensor
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax


class PromptTemplate:
    TEMPLATES = {
        "basic": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "instruction": "Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\n{choices_formatted}\n\nSelect the letter:",
        "simple": "{context}\n{question}\n{choices_formatted}\nThe answer is",
    }

    def __init__(
        self,
        template_name: str = "basic",
        custom_template: Optional[str] = None,
        choice_format: str = "letter",
    ):
        self.template = (
            custom_template
            if custom_template
            else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])
        )
        self.choice_format = choice_format

    def _format_choices(self, choices: List[str]) -> str:
        labels = (
            ["A", "B", "C", "D", "E", "F", "G", "H"]
            if self.choice_format == "letter"
            else [str(i + 1) for i in range(len(choices))]
        )
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))

    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        return self.template.format(
            context=context,
            question=question,
            choices_formatted=self._format_choices(choices),
            **kwargs,
        )

    def format_with_answer(
        self, context: str, question: str, choices: List[str], answer_idx: int
    ) -> str:
        prompt = self.format(context, question, choices)
        label = (
            chr(ord("A") + answer_idx)
            if self.choice_format == "letter"
            else str(answer_idx + 1)
        )
        return f"{prompt} {label}"


class PromptingPipeline:
    def __init__(
        self,
        model,
        tokenizer,
        template: Optional[PromptTemplate] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device) if hasattr(model, "to") else model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = device
        self._setup_choice_tokens()

    def _setup_choice_tokens(self):
        self.choice_tokens = {}
        for label in ["A", "B", "C", "D"]:
            for prefix in ["", " "]:
                token_ids = self.tokenizer.encode(prefix + label)
                if token_ids:
                    self.choice_tokens[label] = token_ids[-1]
                    break

    @torch.no_grad()
    def predict_single(
        self,
        context: str,
        question: str,
        choices: List[str],
        return_probs: bool = False,
    ):
        self.model.eval()
        prompt = self.template.format(context, question, choices)
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        logits = self.model(input_ids)[:, -1, :]

        choice_labels = ["A", "B", "C", "D"][: len(choices)]
        choice_logits = []
        for label in choice_labels:
            if label in self.choice_tokens:
                choice_logits.append(logits[0, self.choice_tokens[label]].item())
            else:
                choice_logits.append(float("-inf"))

        choice_logits = torch.tensor(choice_logits)
        probs = softmax(choice_logits, dim=-1)
        prediction = probs.argmax().item()

        if return_probs:
            return prediction, probs.tolist()
        return prediction

    @torch.no_grad()
    def predict_batch(
        self, examples: List[Dict[str, Any]], batch_size: int = 8
    ) -> List[int]:
        return [
            self.predict_single(ex["context"], ex["question"], ex["choices"])
            for ex in examples
        ]


class FewShotPromptingPipeline(PromptingPipeline):

    def __init__(
        self,
        model,
        tokenizer,
        template: Optional[PromptTemplate] = None,
        device: str = "cuda",
        few_shot_pool: Optional[List[Dict[str, Any]]] = None,
        k: int = 2,
        context_max_chars: int = 200,
        max_input_tokens: int = 490,
        seed: Optional[int] = 42,
    ):
        super().__init__(model, tokenizer, template or PromptTemplate("basic"), device)
        self.few_shot_pool = few_shot_pool or []
        self.k = k
        self.context_max_chars = context_max_chars
        self.max_input_tokens = max_input_tokens
        self._rng = random.Random(seed)

    def _format_example(self, ex: Dict[str, Any]) -> str:
        ctx = ex["context"]
        if len(ctx) > self.context_max_chars:
            ctx = ctx[: self.context_max_chars] + "..."
        block = self.template.format(ctx, ex["question"], ex["choices"])
        answer_label = chr(ord("A") + ex["answer"])
        return f"{block} {answer_label}"

    def _build_prefix(self) -> str:
        if not self.few_shot_pool or self.k == 0:
            return ""
        samples = self._rng.sample(
            self.few_shot_pool, min(self.k, len(self.few_shot_pool))
        )
        return "Here are some examples you can refer to:""\n\n".join(self._format_example(ex) for ex in samples) + "\n\nHere is the question:\n\n"

    @torch.no_grad()
    def predict_single(
        self,
        context: str,
        question: str,
        choices: List[str],
        return_probs: bool = False,
    ):
        self.model.eval()

        ctx_short = context
        if len(ctx_short) > self.context_max_chars:
            ctx_short = ctx_short[: self.context_max_chars] + "..."

        prefix = self._build_prefix()
        query = self.template.format(ctx_short, question, choices)
        prompt = prefix + query

        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) > self.max_input_tokens:
            token_ids = token_ids[-self.max_input_tokens :]

        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self.model(input_ids)[:, -1, :]

        choice_labels = ["A", "B", "C", "D"][: len(choices)]
        choice_logits = [
            logits[0, self.choice_tokens[l]].item()
            if l in self.choice_tokens
            else float("-inf")
            for l in choice_labels
        ]

        choice_logits = torch.tensor(choice_logits)
        probs = softmax(choice_logits, dim=-1)
        prediction = probs.argmax().item()

        if return_probs:
            return prediction, probs.tolist()
        return prediction


def evaluate_prompting(
    pipeline, examples: List[Dict[str, Any]], batch_size: int = 8
) -> Dict[str, Any]:
    predictions = pipeline.predict_batch(examples, batch_size)
    correct = sum(
        1
        for p, ex in zip(predictions, examples)
        if ex.get("answer", -1) >= 0 and p == ex["answer"]
    )
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "predictions": predictions,
    }
