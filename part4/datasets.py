"""
Dataset classes for pre-training and fine-tuning.
Example submission.
"""

import json
import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm
from part4.prompting import PromptTemplate, FewShotPromptingPipeline


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
        self.max_length = max_length
        self.num_choices = num_choices

        self._prompting_input_ids: List[List[int]] = []
        self._prompting_labels: List[int] = []
        self._classification_input_ids: List[List[List[int]]] = []
        self._classification_attention_masks: List[List[List[int]]] = []
        self._classification_labels: List[int] = []
        template = PromptTemplate()
        for example in tqdm(data, desc="Tokenizing QA data", leave=False):
            context  = example["context"]
            question = example["question"]
            choices  = example["choices"]
            answer   = example.get("answer", -1)

            all_ids, all_masks = [], []
            for choice in choices:
                text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {choice}"
                ids = tokenizer.encode(text)
                if len(ids) > max_length:
                    ids = ids[:max_length]
                pad = max_length - len(ids)
                all_ids.append(ids + [0] * pad)
                all_masks.append([1] * len(ids) + [0] * pad)
            text = template.format_with_answer(context=context, question=question, choices=choices,answer_idx=answer)
            ids = tokenizer.encode(text[:-1])
            if len(ids) > max_length:
                    ids = ids[:max_length]
            pad = max_length - len(ids)
            self._prompting_input_ids.append(ids + [0] * pad)
            self._prompting_labels.append(tokenizer.encode(text[-1])[0])
            self._classification_input_ids.append(all_ids)
            self._classification_attention_masks.append(all_masks)
            self._classification_labels.append(answer)

        print(f"QA dataset ready: {len(self._classification_labels):,} examples "
              f"(max_length={max_length}, choices={num_choices})")

    def __len__(self) -> int:
        return len(self._classification_labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "prompting_input_ids":            torch.tensor(self._prompting_input_ids[idx],            dtype=torch.long),
            "prompting_labels":               torch.tensor(self._prompting_labels[idx],               dtype=torch.long),
            "classification_input_ids":       torch.tensor(self._classification_input_ids[idx],       dtype=torch.long),
            "classification_attention_masks": torch.tensor(self._classification_attention_masks[idx], dtype=torch.long),
            "classification_labels":          torch.tensor(self._classification_labels[idx],          dtype=torch.long),
        }

    @classmethod
    def from_json(
        cls, file_path: str | Path, tokenizer, **kwargs
    ) -> "MultipleChoiceQADataset":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, tokenizer, **kwargs)


class _PromptFormatter(FewShotPromptingPipeline):

    def __init__(self, tokenizer, template, few_shot_pool, k, context_max_chars, seed):
        self.tokenizer = tokenizer
        self.template = template
        self.few_shot_pool = few_shot_pool
        self.k = k
        self.context_max_chars = context_max_chars
        self.max_input_tokens = 1 << 7 
        self._rng = random.Random(seed)
        self._setup_choice_tokens()  


class PromptingFineTuneDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        template: Optional[PromptTemplate] = None,
        few_shot_pool: Optional[List[Dict[str, Any]]] = None,
        beta: float = 0.5,
        k: int = 2,
        context_max_chars: int = 200,
        max_length: int = 512,
        seed: int = 42,
    ):
        template = template or PromptTemplate("basic")
        few_shot_pool = few_shot_pool or []

        formatter = _PromptFormatter(
            tokenizer=tokenizer,
            template=template,
            few_shot_pool=few_shot_pool,
            k=k,
            context_max_chars=context_max_chars,
            seed=seed,
        )

        rng = random.Random(seed)
        self._input_ids_list: List[List[int]] = []
        self._seq_lens: List[int] = []
        self._labels: List[int] = []

        n_fs = 0
        for ex in tqdm(data, desc="Building PromptingFineTuneDataset", leave=False):
            use_few_shot = rng.random() < beta

            prefix = formatter._build_prefix() if (use_few_shot and few_shot_pool and k > 0) else ""
            if use_few_shot and few_shot_pool and k > 0:
                n_fs += 1

            ctx = ex["context"]
            if len(ctx) > context_max_chars:
                ctx = ctx[:context_max_chars] + "..."
            query = template.format(ctx, ex["question"], ex["choices"])
            prompt = prefix + query

            token_ids = tokenizer.encode(prompt)
            if len(token_ids) > max_length:
                token_ids = token_ids[-max_length:]

            seq_len = len(token_ids)
            input_ids = token_ids + [0] * (max_length - seq_len)

            answer_letter = chr(ord("A") + ex["answer"])
            label = formatter.choice_tokens.get(
                answer_letter, next(iter(formatter.choice_tokens.values()))
            )

            self._input_ids_list.append(input_ids)
            self._seq_lens.append(seq_len)
            self._labels.append(label)

        print(
            f"PromptingFineTuneDataset ready: {len(data):,} examples "
            f"({n_fs:,} few-shot / {len(data) - n_fs:,} zero-shot, "
            f"beta={beta}, k={k}, max_length={max_length})"
        )

    def __len__(self) -> int:
        return len(self._input_ids_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self._input_ids_list[idx], dtype=torch.long),
            "seq_len":   torch.tensor(self._seq_lens[idx],        dtype=torch.long),
            "label":     torch.tensor(self._labels[idx],          dtype=torch.long),
        }


def create_prompting_finetune_dataloader(
    data: List[Dict[str, Any]],
    tokenizer,
    template: Optional[PromptTemplate] = None,
    few_shot_pool: Optional[List[Dict[str, Any]]] = None,
    beta: float = 0.5,
    k: int = 2,
    context_max_chars: int = 200,
    max_length: int = 512,
    seed: int = 42,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = PromptingFineTuneDataset(
        data=data,
        tokenizer=tokenizer,
        template=template,
        few_shot_pool=few_shot_pool,
        beta=beta,
        k=k,
        context_max_chars=context_max_chars,
        max_length=max_length,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


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
    num_workers=4,
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
        persistent_workers=num_workers > 0,
    )
