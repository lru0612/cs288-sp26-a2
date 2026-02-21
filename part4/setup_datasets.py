#!/usr/bin/env python3
"""
Download and prepare datasets for Part 4.

Datasets:
- TinyStories: ~2.1M short children's stories for pretraining
- Wikipedia (20220301.en): English Wikipedia articles for pretraining
- SQuAD v1.1: ~100k QA examples for fine-tuning

Usage:
    python part4/setup_datasets.py                  # Download all datasets
    python part4/setup_datasets.py --no-wikipedia   # Skip Wikipedia
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Remove current directory from path to avoid importing local datasets.py
# instead of the HuggingFace datasets library
_this_dir = str(Path(__file__).parent)
if _this_dir in sys.path:
    sys.path.remove(_this_dir)

# Try to import datasets, install if needed
try:
    import datasets as hf_datasets

    load_dataset = hf_datasets.load_dataset
except ImportError:
    print("Installing 'datasets' library...")
    os.system("pip install datasets")
    import datasets as hf_datasets

    load_dataset = hf_datasets.load_dataset

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)


def download_tinystories():
    """Download TinyStories dataset for pretraining."""
    print("=" * 60)
    print("Downloading TinyStories dataset...")
    print("=" * 60)

    # Load TinyStories from HuggingFace
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    print(f"Total stories: {len(dataset):,}")

    # Save as text file with <|endoftext|> separator
    output_path = FIXTURES_DIR / "tinystories_full.txt"
    if output_path.exists():
        print(f"Found existing file, skipping download.")
        return output_path

    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            story = example["text"].strip()
            f.write(story)
            f.write("\n<|endoftext|>\n")

            if (i + 1) % 100000 == 0:
                print(f"  Processed {i + 1:,} stories...")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")

    # Also create a smaller subset (100k stories) with train/val split
    subset_path = FIXTURES_DIR / "tinystories_100k.txt"
    subset_val_path = FIXTURES_DIR / "tinystories_100k_val.txt"

    with open(subset_path, "w", encoding="utf-8") as f_train, \
         open(subset_val_path, "w", encoding="utf-8") as f_val:
        for i, example in enumerate(dataset):
            if i >= 100000:
                break
            story = example["text"].strip()
            text = story + "\n<|endoftext|>\n"
            if i < 95000:
                f_train.write(text)
            else:
                f_val.write(text)

    print(f"Also created 100k train subset : {subset_path}  (95k stories)")
    print(f"Also created 100k val   subset : {subset_val_path}  (5k stories)")

    return output_path


def download_wikipedia(
    num_train: int = 100000,
    num_val: int = 5000,
    language: str = "20231101.en",
):
    """
    Download Wikipedia dataset for pretraining.

    Args:
        num_train: Number of articles to use for training
        num_val:   Number of articles to use for validation
        language:  Wikipedia dump version (default: 20220301.en)

    Returns:
        (train_path, val_path)
    """
    print("\n" + "=" * 60)
    print(f"Downloading Wikipedia ({language}) dataset...")
    print("=" * 60)

    train_path = FIXTURES_DIR / f"wikipedia_{num_train // 1000}k.txt"
    val_path   = FIXTURES_DIR / f"wikipedia_{num_train // 1000}k_val.txt"

    if train_path.exists() and val_path.exists():
        train_mb = train_path.stat().st_size / (1024 * 1024)
        val_mb   = val_path.stat().st_size / (1024 * 1024)
        print(f"Found existing files, skipping download.")
        print(f"  Train: {train_path}  ({train_mb:.1f} MB)")
        print(f"  Val:   {val_path}  ({val_mb:.1f} MB)")
        return train_path, val_path

    dataset = load_dataset("wikimedia/wikipedia", language, split="train")
    total = len(dataset)
    need  = num_train + num_val
    print(f"Total Wikipedia articles: {total:,}")
    print(f"Using: {num_train:,} train + {num_val:,} val = {need:,} articles")

    if need > total:
        raise ValueError(
            f"Requested {need:,} articles but dataset only has {total:,}."
        )

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path,   "w", encoding="utf-8") as f_val:
        for i, example in enumerate(dataset):
            if i >= need:
                break
            article = example["text"].strip()
            if not article:
                continue
            text = article + "\n<|endoftext|>\n"
            if i < num_train:
                f_train.write(text)
            else:
                f_val.write(text)

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1:,} articles...")

    train_mb = train_path.stat().st_size / (1024 * 1024)
    val_mb   = val_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved:")
    print(f"  Train: {train_path}  ({train_mb:.1f} MB, {num_train:,} articles)")
    print(f"  Val:   {val_path}  ({val_mb:.1f} MB, {num_val:,} articles)")

    return train_path, val_path


def download_squad():
    """Download SQuAD v1.1 dataset for QA fine-tuning."""
    print("\n" + "=" * 60)
    print("Downloading SQuAD v1.1 dataset...")
    print("=" * 60)

    # Load SQuAD from HuggingFace
    dataset = load_dataset("squad", split="train")
    val_dataset = load_dataset("squad", split="validation")

    print(f"Training examples: {len(dataset):,}")
    print(f"Validation examples: {len(val_dataset):,}")

    def convert_to_multiple_choice(examples, num_examples=None, num_distractors=3):
        """
        Convert extractive QA to multiple choice format.

        For each question:
        - Correct answer: the actual answer from the dataset
        - Distractors: answers from other questions in the same context
        """
        import random

        # Group by context to find distractors
        context_answers = {}
        for ex in examples:
            ctx_id = ex["context"][:100]  # Use first 100 chars as key
            if ctx_id not in context_answers:
                context_answers[ctx_id] = []
            context_answers[ctx_id].append(ex["answers"]["text"][0])

        # Get all unique answers for random distractors
        all_answers = list(
            set(ex["answers"]["text"][0] for ex in examples if ex["answers"]["text"])
        )

        converted = []
        for i, ex in enumerate(examples):
            if num_examples and i >= num_examples:
                break

            if not ex["answers"]["text"]:
                continue

            correct_answer = ex["answers"]["text"][0]

            # Get distractors (other answers, preferring same context)
            ctx_id = ex["context"][:100]
            same_ctx_answers = [
                a for a in context_answers.get(ctx_id, []) if a != correct_answer
            ]
            other_answers = [
                a
                for a in all_answers
                if a != correct_answer and a not in same_ctx_answers
            ]

            # Select distractors
            distractors = []
            if same_ctx_answers:
                distractors.extend(
                    random.sample(same_ctx_answers, min(1, len(same_ctx_answers)))
                )

            remaining = num_distractors - len(distractors)
            if remaining > 0 and other_answers:
                distractors.extend(
                    random.sample(other_answers, min(remaining, len(other_answers)))
                )

            # Pad with generic distractors if needed
            generic = ["Unknown", "Not mentioned", "Cannot determine"]
            while len(distractors) < num_distractors:
                distractors.append(random.choice(generic))

            # Create choices and shuffle
            choices = [correct_answer] + distractors[:num_distractors]
            answer_idx = 0  # Correct answer is first

            # Shuffle
            indices = list(range(len(choices)))
            random.shuffle(indices)
            choices = [choices[i] for i in indices]
            answer_idx = indices.index(0)

            converted.append(
                {
                    "context": ex["context"],
                    "question": ex["question"],
                    "choices": choices,
                    "answer": answer_idx,
                    "id": ex["id"],
                }
            )

        return converted

    # Convert datasets
    print("\nConverting to multiple-choice format...")

    # Training set: use 10k examples (manageable for assignment)
    train_mc = convert_to_multiple_choice(dataset, num_examples=10000)

    # Validation: use 2k examples
    val_mc = convert_to_multiple_choice(val_dataset, num_examples=2000)

    # Test: use 1k examples (subset of validation for hidden test)
    test_mc = convert_to_multiple_choice(val_dataset, num_examples=1000)

    # Save
    train_path = FIXTURES_DIR / "squad_train.json"
    val_path = FIXTURES_DIR / "squad_dev.json"
    test_path = FIXTURES_DIR / "squad_test.json"

    with open(train_path, "w") as f:
        json.dump(train_mc, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val_mc, f, indent=2)

    with open(test_path, "w") as f:
        json.dump(test_mc, f, indent=2)

    print(f"\nSaved:")
    print(f"  Training:   {train_path} ({len(train_mc):,} examples)")
    print(f"  Validation: {val_path} ({len(val_mc):,} examples)")
    print(f"  Test:       {test_path} ({len(test_mc):,} examples)")

    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser(description="CS288 Part 4 - Dataset Setup")
    parser.add_argument(
        "--no-wikipedia", action="store_true",
        help="Skip downloading Wikipedia (saves time/disk space)"
    )
    parser.add_argument(
        "--wiki-train", type=int, default=100000,
        help="Number of Wikipedia articles for training"
    )
    parser.add_argument(
        "--wiki-val", type=int, default=1000,
        help="Number of Wikipedia articles for validation"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("CS288 Part 4 - Dataset Setup")
    print("=" * 60 + "\n")

    # Download TinyStories
    tinystories_path = download_tinystories()

    # Download Wikipedia (optional)
    wikipedia_paths = None
    if not args.no_wikipedia:
        wikipedia_paths = download_wikipedia(
            num_train=args.wiki_train,
            num_val=args.wiki_val,
        )

    # Download SQuAD
    # squad_paths = download_squad()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nDatasets ready in:", FIXTURES_DIR)
    print("\nRecommended usage:")
    print("  - Pretraining (TinyStories only):  tinystories_100k.txt")
    if wikipedia_paths:
        print(f"  - Pretraining (Wikipedia only):    {wikipedia_paths[0].name}")
        print("  - Pretraining (Mixed):             use MixedPretrainingDataset")
        print("    e.g. MixedPretrainingDataset(")
        print("           tinystories_path='fixtures/tinystories_100k.txt',")
        print(f"           wikipedia_path='fixtures/{wikipedia_paths[0].name}',")
        print("           tinystories_weight=0.7, wikipedia_weight=0.3)")
    print("  - Fine-tuning: squad_train.json (10k examples)")
    print("  - Validation:  squad_dev.json (2k examples)")
    print("  - Testing:     squad_test.json (1k examples)")


if __name__ == "__main__":
    main()
