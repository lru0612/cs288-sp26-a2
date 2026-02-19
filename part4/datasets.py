"""
Dataset classes for pre-training and fine-tuning.
Example submission.
"""

import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union


class PretrainingDataset(Dataset):
    def __init__(
        self,
        file_path: str | Path,
        tokenizer,
        max_length: int = 256,
        stride: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length

        cache_path = Path(str(file_path) + ".token_cache.pkl")
        if cache_path.exists():
            print(f"Loading tokenized cache from {cache_path}...")
            with open(cache_path, "rb") as f:
                self.token_ids = pickle.load(f)
            print(f"Cache loaded! ({len(self.token_ids):,} tokens)")
        else:
            print(f"Tokenizing corpus...")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.token_ids = tokenizer.encode(text, show_progress=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.token_ids, f)
            print(f"Tokenization done! ({len(self.token_ids):,} tokens, cache saved to {cache_path})")
        if len(self.token_ids) <= max_length:
            self.num_sequences = 1
        else:
            self.num_sequences = (len(self.token_ids) - max_length) // self.stride + 1

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx * self.stride
        end_idx = min(start_idx + self.max_length + 1, len(self.token_ids))
        sequence = self.token_ids[start_idx:end_idx]
        if len(sequence) < self.max_length + 1:
            sequence = sequence + [0] * (self.max_length + 1 - len(sequence))
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class ConcatPretrainingDataset(Dataset):
    def __init__(
        self,
        file_paths: Union[List[Union[str, Path]], str, Path],
        tokenizer,
        max_length: int = 512,
        stride: Optional[int] = None,
    ):
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        self.max_length = max_length
        self.stride = stride or max_length

        self._all_tokens: List[List[int]] = []
        self._seq_counts: List[int] = []
        self._offsets: List[int] = [] 

        total_seqs = 0
        for fp in file_paths:
            tokens = self._load_tokens(fp, tokenizer)
            n_seqs = self._count_sequences(tokens)
            self._all_tokens.append(tokens)
            self._seq_counts.append(n_seqs)
            self._offsets.append(total_seqs)
            total_seqs += n_seqs

        self._total_seqs = total_seqs

        print(
            f"\nConcatPretrainingDataset ready:\n"
            + "\n".join(
                f"  [{i}] {Path(fp).name}: {len(tok):,} tokens → {cnt:,} sequences"
                for i, (fp, tok, cnt) in enumerate(
                    zip(file_paths, self._all_tokens, self._seq_counts)
                )
            )
            + f"\n  Total: {self._total_seqs:,} sequences"
        )

    def _load_tokens(self, file_path: Union[str, Path], tokenizer) -> List[int]:
        file_path = Path(file_path)
        cache_path = Path(str(file_path) + ".token_cache.pkl")
        if cache_path.exists():
            print(f"[{file_path.name}] Loading cache from {cache_path} ...")
            with open(cache_path, "rb") as f:
                tokens = pickle.load(f)
            print(f"[{file_path.name}] Cache loaded! ({len(tokens):,} tokens)")
            return tokens

        print(f"[{file_path.name}] Tokenizing ...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text, show_progress=True)
        with open(cache_path, "wb") as f:
            pickle.dump(tokens, f)
        print(f"[{file_path.name}] Done! ({len(tokens):,} tokens, cache → {cache_path})")
        return tokens

    def _count_sequences(self, token_ids: List[int]) -> int:
        if len(token_ids) <= self.max_length:
            return 1
        return (len(token_ids) - self.max_length) // self.stride + 1

    def _get_sequence(self, token_ids: List[int], local_idx: int) -> Dict[str, torch.Tensor]:
        start = local_idx * self.stride
        end   = min(start + self.max_length + 1, len(token_ids))
        seq   = token_ids[start:end]
        if len(seq) < self.max_length + 1:
            seq = seq + [0] * (self.max_length + 1 - len(seq))
        return {
            "input_ids": torch.tensor(seq[:-1], dtype=torch.long),
            "labels":    torch.tensor(seq[1:],  dtype=torch.long),
        }

    def __len__(self) -> int:
        return self._total_seqs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        local_idx = idx - self._offsets[lo]
        return self._get_sequence(self._all_tokens[lo], local_idx)


class MultipleChoiceQADataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 256,
        num_choices: int = 4,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_choices = num_choices

    def __len__(self) -> int:
        return len(self.data)

    def _format_choice_input(self, context: str, question: str, choice: str) -> str:
        return f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {choice}"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        context = example["context"]
        question = example["question"]
        choices = example["choices"]
        answer = example.get("answer", -1)

        all_input_ids = []
        all_attention_masks = []

        for choice in choices:
            text = self._format_choice_input(context, question, choice)
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[: self.max_length]
            attention_mask = [1] * len(token_ids)
            padding_length = self.max_length - len(token_ids)
            token_ids = token_ids + [0] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            all_input_ids.append(token_ids)
            all_attention_masks.append(attention_mask)

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
            "labels": torch.tensor(answer, dtype=torch.long),
        }

    @classmethod
    def from_json(
        cls, file_path: str | Path, tokenizer, **kwargs
    ) -> "MultipleChoiceQADataset":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, tokenizer, **kwargs)


def create_pretraining_dataloader(
    file_path,
    tokenizer,
    batch_size=8,
    max_length=256,
    stride=None,
    shuffle=True,
    num_workers=4,
):
    dataset = PretrainingDataset(file_path, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def create_qa_dataloader(
    data,
    tokenizer,
    batch_size=4,
    max_length=256,
    num_choices=4,
    shuffle=True,
    num_workers=0,
):
    if isinstance(data, (str, Path)):
        dataset = MultipleChoiceQADataset.from_json(
            data, tokenizer, max_length=max_length, num_choices=num_choices
        )
    else:
        dataset = MultipleChoiceQADataset(
            data, tokenizer, max_length=max_length, num_choices=num_choices
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
