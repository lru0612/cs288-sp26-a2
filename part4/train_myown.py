#!/usr/bin/env python3
"""
Part 4 Baseline Training Script

This script demonstrates how to:
1. Train a BPE tokenizer on TinyStories
2. Pretrain a Transformer LM for next-token prediction
3. Fine-tune the model for multiple-choice QA
4. Evaluate using both prompting and fine-tuning approaches

Students can use this as a reference for their implementations.

Usage:
    # First, download datasets
    python part4/setup_datasets.py

    # Then run training (use --quick for testing)
    python part4/train_baseline.py --quick      # Quick test (~2 min)
    python part4/train_baseline.py              # Full training (~30 min on GPU)
"""

import argparse
import json
import pickle
import sys
import tempfile
import torch
from pathlib import Path

import wandb
import time
# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from part1.train_bpe import train_bpe
from part1.tokenizer import get_tokenizer
from part2.model import TransformerLM
from part3.nn_utils import cross_entropy, gradient_clipping
from torch.utils.data import DataLoader
from part4.datasets import PretrainingDataset, ConcatPretrainingDataset, create_pretraining_dataloader, create_qa_dataloader
from part4.sampling import generate_text
from part4.qa_model import TransformerForMultipleChoice, evaluate_qa_model
from part4.prompting import PromptTemplate, PromptingPipeline, FewShotPromptingPipeline, evaluate_prompting
from part4.trainer import Trainer, TrainingConfig, create_qa_loss_fn

# =============================================================================
# Configuration
# =============================================================================

CONFIGS = {
        "bpe_data": [
            Path(__file__).parent / "fixtures/tinystories_100k.txt",
            Path(__file__).parent / "fixtures/wikipedia_10k.txt",
        ],
        "pretrain_data": [
            Path(__file__).parent / "fixtures/tinystories_100k.txt",
            Path(__file__).parent / "fixtures/wikipedia_100k.txt",
        ],
        "pretrain_val_data": [
            Path(__file__).parent / "fixtures/tinystories_100k_val.txt",
            Path(__file__).parent / "fixtures/wikipedia_100k_val.txt",
        ],
        "qa_train": Path(__file__).parent / "fixtures/squad_train.json",
        "qa_dev": Path(__file__).parent / "fixtures/squad_dev.json",
        "vocab_size": 10240,
        "d_model": 512,
        "num_layers": 10,
        "num_heads": 16,
        "d_ff": 2048,
        "context_length": 512,
        "pretrain_epochs": 5,
        "finetune_epochs": 15,
        "batch_size": 16,
        "val_per_steps": 200,
        "pretrain_patience": 2000,   
        "finetune_val_per_steps": 50,  
        "finetune_patience": 500,      
        "lr": 1e-4,
    }



# =============================================================================
# Step 1: Train BPE Tokenizer
# =============================================================================


def train_tokenizer(pretrain_data, vocab_size: int) -> tuple:
    """
    Train a BPE tokenizer on the pretraining corpus.

    Args:
        pretrain_data: Path (or list of Paths) to training text file(s)
        vocab_size: Target vocabulary size

    Returns:
        (tokenizer, vocab, merges)
    """
    print("\n" + "=" * 60)
    print("STEP 1: Training BPE Tokenizer")
    print("=" * 60)

    special_tokens = ["<|endoftext|>", "<|pad|>"]
    models_dir = Path(__file__).parent / "models"
    tokenizer_path = models_dir / "tokenizer.pkl"

    if tokenizer_path.exists():
        print(f"Found existing tokenizer at {tokenizer_path}, loading...")
        with open(tokenizer_path, "rb") as f:
            saved = pickle.load(f)
        vocab = saved["vocab"]
        merges = saved["merges"]
        tokenizer = get_tokenizer(vocab, merges, special_tokens)
        print(f"Tokenizer loaded! Vocab size: {len(vocab)}, Merges: {len(merges)}")
        return tokenizer, vocab, merges

    _tmp_file = None
    if isinstance(pretrain_data, list):
        print(f"Input: {[str(p) for p in pretrain_data]} ")
        _tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        for fp in pretrain_data:
            fp = Path(fp)
            if fp.exists():
                _tmp.write(fp.read_text(encoding="utf-8"))
                _tmp.write("\n")
        _tmp.flush()
        _tmp.close()
        bpe_input_path = Path(_tmp.name)
        _tmp_file = bpe_input_path
    else:
        bpe_input_path = Path(pretrain_data)
        print(f"Input: {bpe_input_path}")

    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")

    # Train BPE
    vocab, merges = train_bpe(
        input_path=bpe_input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    if _tmp_file is not None and _tmp_file.exists():
        _tmp_file.unlink()

    # Create tokenizer
    tokenizer = get_tokenizer(vocab, merges, special_tokens)

    # Test
    test_text = "Once upon a time, there was a little girl."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)

    print(f"\nTokenizer trained!")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Merges: {len(merges)}")
    print(f"\nTest encoding:")
    print(f"  Input:   '{test_text}'")
    print(f"  Tokens:  {len(tokens)} tokens")
    print(f"  Decoded: '{decoded}'")

    models_dir.mkdir(parents=True, exist_ok=True)
    with open(tokenizer_path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    print(f"\nTokenizer saved to {tokenizer_path}")

    return tokenizer, vocab, merges


# =============================================================================
# Step 2: Pretrain Language Model
# =============================================================================


def pretrain_lm(
    tokenizer,
    config: dict,
    device: str = "cpu",
    use_wandb: bool = False,
) -> TransformerLM:
    """
    Pretrain a Transformer language model on TinyStories.

    The model learns to predict the next token given previous tokens.
    This gives it general language understanding before fine-tuning.

    Args:
        tokenizer: Trained BPE tokenizer
        config: Model and training configuration
        device: Device to train on

    Returns:
        Pretrained TransformerLM
    """
    print("\n" + "=" * 60)
    print("STEP 2: Pretraining Language Model")
    print("=" * 60)

    # Create model
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel architecture:")
    print(f"  d_model: {config['d_model']}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  d_ff: {config['d_ff']}")
    print(f"  context_length: {config['context_length']}")
    print(f"  Parameters: {num_params:,}")

    # Create dataloaders
    _dl_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    _ds_kwargs = dict(
        tokenizer=tokenizer,
        max_length=config["context_length"],
        stride=config["context_length"] // 2,
    )

    pretrain_data = config["pretrain_data"]
    if isinstance(pretrain_data, list):
        train_dataset = ConcatPretrainingDataset(file_paths=pretrain_data, **_ds_kwargs)
    else:
        train_dataset = PretrainingDataset(file_path=pretrain_data, **_ds_kwargs)
    dataloader = DataLoader(train_dataset, shuffle=True, **_dl_kwargs)

    val_dataloader = None
    val_paths = config.get("pretrain_val_data")
    if val_paths:
        if isinstance(val_paths, list):
            existing = [p for p in val_paths if Path(p).exists()]
            if existing:
                val_dataset = ConcatPretrainingDataset(file_paths=existing, **_ds_kwargs)
                val_dataloader = DataLoader(val_dataset, shuffle=False, **_dl_kwargs)
        elif Path(val_paths).exists():
            val_dataset = PretrainingDataset(file_path=val_paths, **_ds_kwargs)
            val_dataloader = DataLoader(val_dataset, shuffle=False, **_dl_kwargs)

    print(f"\nTraining data:")
    print(f"  Source(s): {pretrain_data}")
    print(f"  Train sequences: {len(train_dataset)}")
    if val_dataloader:
        print(f"  Val   sequences: {len(val_dataset)}  (source: {val_paths})")
    print(f"  Train batches/epoch: {len(dataloader)}" + (f"  |  Val batches: {len(val_dataloader)}" if val_dataloader else ""))

    # Training config
    train_config = TrainingConfig(
        num_epochs=config["pretrain_epochs"],
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_steps=min(100, len(dataloader) // 5),
        max_grad_norm=1.0,
        device=device,
        log_interval=max(1, len(dataloader) // 5),
        use_amp=device == "cuda",
        val_per_steps=config["val_per_steps"],
        patience=config.get("pretrain_patience"),  # step 级早停，None 表示不启用
        use_wandb=use_wandb,
    )

    # Train
    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataloader=dataloader,
        val_dataloader=val_dataloader,
    )

    print(f"\nTraining for {config['pretrain_epochs']} epoch(s)...")
    results = trainer.train()

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%m%d-%H%M")
    pretrain_ckpt = models_dir / f"pretrained_lm_{ts}.pt"
    torch.save(model.state_dict(), pretrain_ckpt)
    print(f"\nPretrained model saved to {pretrain_ckpt}")

    # Test generation
    print("\nGeneration test:")
    for prompt in ["Once upon a time", "The little dog"]:
        generated = generate_text(
            model, tokenizer, prompt, max_new_tokens=30, method="greedy"
        )
        print(f"  '{prompt}' -> '{generated[:80]}...'")

    return model


# =============================================================================
# Step 3: Evaluate Zero-Shot Prompting
# =============================================================================


def evaluate_prompting(
    model: TransformerLM,
    tokenizer,
    qa_dev_path: Path,
    device: str = "cpu",
) -> dict:
    """
    Evaluate the pretrained model using zero-shot prompting.

    This tests if the model can answer questions without any fine-tuning,
    just by predicting which answer token (A, B, C, D) is most likely.

    Args:
        model: Pretrained TransformerLM
        tokenizer: Tokenizer
        qa_dev_path: Path to validation QA data
        device: Device

    Returns:
        Evaluation results dict
    """
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating Prompting (on fine-tuned model)")
    print("=" * 60)

    # Load data
    with open(qa_dev_path) as f:
        dev_data = json.load(f)

    print(f"\nValidation examples: {len(dev_data)}")

    qa_train_path = Path(__file__).parent / "fixtures/squad_train.json"
    few_shot_pool = []
    if qa_train_path.exists():
        with open(qa_train_path) as f:
            few_shot_pool = json.load(f)
        print(f"  Few-shot pool: {len(few_shot_pool):,} examples from {qa_train_path.name}")

    template = PromptTemplate(template_name="basic")
    pipeline = FewShotPromptingPipeline(
        model=model,
        tokenizer=tokenizer,
        template=template,
        device=device,
        few_shot_pool=few_shot_pool,
        k=2,                  
        context_max_chars=200, 
        max_input_tokens=490, 
        seed=42,
    )

    # Evaluate
    from part4.prompting import evaluate_prompting as eval_prompt

    results = eval_prompt(pipeline, dev_data)

    print(f"\nPrompting accuracy (on fine-tuned model): {results['accuracy']:.2%}")
    print(f"Random baseline: 25.00%")

    return results


# =============================================================================
# Step 4: Fine-tune for QA
# =============================================================================


def finetune_qa(
    pretrained_model: TransformerLM,
    tokenizer,
    config: dict,
    device: str = "cpu",
    use_wandb: bool = False,
) -> TransformerForMultipleChoice:
    """
    Fine-tune the pretrained model for multiple-choice QA.

    We add a classification head and train the entire model to select
    the correct answer from 4 choices.

    Args:
        pretrained_model: Pretrained TransformerLM
        tokenizer: Tokenizer
        config: Training configuration
        device: Device

    Returns:
        Fine-tuned QA model
    """
    print("\n" + "=" * 60)
    print("STEP 3: Fine-tuning for Multiple-Choice QA")
    print("=" * 60)

    # Create QA model (wraps the LM with a classification head)
    qa_model = TransformerForMultipleChoice(
        transformer_lm=pretrained_model,
        hidden_size=pretrained_model.d_model,
        num_choices=4,
        pooling="last",  # Use last token representation
        freeze_backbone=False,  # Fine-tune entire model
    ).to(device)

    print(f"\nQA model parameters: {sum(p.numel() for p in qa_model.parameters()):,}")

    # Load training data
    with open(config["qa_train"]) as f:
        train_data = json.load(f)

    _qa_dl_kwargs = dict(
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        num_choices=4,
    )
    train_dataloader = create_qa_dataloader(data=train_data, shuffle=True, **_qa_dl_kwargs)

    # Load validation data for step-level early stopping
    with open(config["qa_dev"]) as f:
        dev_data = json.load(f)
    val_dataloader = create_qa_dataloader(data=dev_data, shuffle=False, **_qa_dl_kwargs)

    print(f"\nTraining data: {config['qa_train']}")
    print(f"Training examples: {len(train_data)}")
    print(f"Batches/epoch: {len(train_dataloader)}")
    print(f"Val examples: {len(dev_data)}")

    # Training config
    train_config = TrainingConfig(
        num_epochs=config["finetune_epochs"],
        learning_rate=config["lr"] / 2,  # Lower LR for fine-tuning
        weight_decay=0.01,
        warmup_steps=min(50, len(train_dataloader) // 5),
        max_grad_norm=1.0,
        device=device,
        log_interval=max(1, len(train_dataloader) // 5),
        use_amp=device == "cuda",
        val_per_steps=config.get("finetune_val_per_steps"),
        patience=config.get("finetune_patience"),  # step 级早停
        use_wandb=use_wandb,
    )

    # Train
    trainer = Trainer(
        model=qa_model,
        config=train_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        compute_loss_fn=create_qa_loss_fn(device),
    )

    print(f"\nFine-tuning for {config['finetune_epochs']} epoch(s)...")
    results = trainer.train()

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%m%d-%H%M")
    finetune_ckpt = models_dir / f"finetuned_qa_{ts}.pt"
    torch.save(qa_model.state_dict(), finetune_ckpt)
    print(f"\nFine-tuned QA model saved to {finetune_ckpt}")

    return qa_model


# =============================================================================
# Step 5: Evaluate Fine-tuned Model
# =============================================================================


def evaluate_finetuned(
    qa_model: TransformerForMultipleChoice,
    tokenizer,
    config: dict,
    device: str = "cpu",
) -> dict:
    """
    Evaluate the fine-tuned QA model.

    Args:
        qa_model: Fine-tuned QA model
        tokenizer: Tokenizer
        config: Configuration
        device: Device

    Returns:
        Evaluation results
    """
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating Fine-tuned Model")
    print("=" * 60)

    # Load validation data
    with open(config["qa_dev"]) as f:
        dev_data = json.load(f)

    dev_dataloader = create_qa_dataloader(
        data=dev_data,
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["context_length"],
        num_choices=4,
        shuffle=False,
    )

    print(f"\nValidation examples: {len(dev_data)}")

    # Evaluate
    results = evaluate_qa_model(qa_model, dev_dataloader, device)

    print(f"\nFine-tuned model accuracy: {results['accuracy']:.2%}")
    print(f"Random baseline: 25.00%")

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Part 4 Myown Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (auto-detect if not set)"
    )
    args = parser.parse_args()


    config = CONFIGS
    config_name = "myown"
    use_wandb=True
    if use_wandb:
        pretrain_data = config["pretrain_data"]
        pretrain_data_str = (
            [str(p) for p in pretrain_data]
            if isinstance(pretrain_data, list)
            else str(pretrain_data)
        )
        wandb.init(
                project="cs288-proj2",
                name=f"MyLM-{time.strftime('%Y%m%d-%H%M')}",
                config={
                    **{k: v for k, v in config.items()
                       if k not in ("pretrain_data", "pretrain_val_data", "qa_train", "qa_dev")},
                    "pretrain_data": pretrain_data_str,
                    "qa_train": str(config["qa_train"]),
                    "qa_dev": str(config["qa_dev"]),
                    "config_name": config_name,
                },
            )
        print(f"W&B run initialized: {wandb.run.url}")


    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("CS288 Part 4 - My Training")
    print("=" * 60)
    print(f"Device: {device}")

    # Step 1: Train tokenizer
    # Use bpe_data if specified (faster for large configs), otherwise use pretrain_data
    bpe_data = config.get("bpe_data", config["pretrain_data"])
    tokenizer, vocab, merges = train_tokenizer(bpe_data, config["vocab_size"])

    # Step 2: Pretrain LM
    pretrained_model = pretrain_lm(tokenizer, config, device, use_wandb=use_wandb)

    if device == "cuda":
        torch.cuda.empty_cache()

    # Step 3: Fine-tune for QA
    qa_model = finetune_qa(pretrained_model, tokenizer, config, device, use_wandb=use_wandb)

    # Step 4: Evaluate prompting on fine-tuned model
    # Use the fine-tuned backbone (qa_model.transformer) for prompting
    prompting_results = evaluate_prompting(
        qa_model.transformer, tokenizer, config["qa_dev"], device
    )

    # Step 5: Evaluate fine-tuned model (classification head)
    finetuned_results = evaluate_finetuned(qa_model, tokenizer, config, device)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Model parameters: {sum(p.numel() for p in pretrained_model.parameters()):,}"
    )
    print(f"\nResults (both on fine-tuned model):")
    print(f"  Prompting approach:    {prompting_results['accuracy']:.2%}")
    print(f"  Classification head:   {finetuned_results['accuracy']:.2%}")
    print(f"  Random baseline:       25.00%")

    # Calculate improvement (prompting should beat finetuned for full prompting score)
    prompting_boost = prompting_results["accuracy"] - finetuned_results["accuracy"]
    print(f"\n  Prompting boost over fine-tuned: {prompting_boost:+.2%}")
    if prompting_boost >= 0.04:
        print(f"  (4%+ boost = full prompting score)")
    elif prompting_boost > 0:
        print(f"  (Need 4% boost for full prompting score)")
    else:
        print(f"  (Prompting should beat fine-tuned model)")

    # Save predictions to JSON files for grading
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Save fine-tuned predictions
    finetuned_output = {
        "predictions": finetuned_results.get("predictions", []),
        "accuracy": finetuned_results["accuracy"],
        "config": config_name,
    }
    finetuned_path = output_dir / "finetuned_predictions.json"
    with open(finetuned_path, "w") as f:
        json.dump(finetuned_output, f, indent=2)

    # Save prompting predictions
    prompting_output = {
        "predictions": prompting_results.get("predictions", []),
        "accuracy": prompting_results["accuracy"],
        "config": config_name,
    }
    prompting_path = output_dir / "prompting_predictions.json"
    with open(prompting_path, "w") as f:
        json.dump(prompting_output, f, indent=2)

    print(f"\nPredictions saved to:")
    print(f"  {finetuned_path}")
    print(f"  {prompting_path}")

    # Print grading info
    print("\n" + "=" * 60)
    print("GRADING RUBRIC")
    print("=" * 60)
    finetuned_score = max(0, min(1, (finetuned_results["accuracy"] - 0.30) / 0.20))
    prompting_score = (
        max(0, min(1, prompting_boost / 0.04)) if prompting_boost > 0 else 0
    )
    total_score = 0.5 * finetuned_score + 0.5 * prompting_score

    print(f"\nFine-tuned score:  {finetuned_score:.0%} (30%=0pts, 50%=full)")
    print(f"Prompting score:   {prompting_score:.0%} (0% boost=0pts, 4% boost=full)")
    print(f"Total Part 4:      {total_score:.0%}")

    print("\nDone!")

    if use_wandb:
        wandb.log(
                {
                    "eval/finetuned_accuracy": finetuned_results["accuracy"],
                    "eval/prompting_accuracy": prompting_results["accuracy"],
                    "eval/prompting_boost": prompting_boost,
                    "eval/finetuned_score": finetuned_score,
                    "eval/prompting_score": prompting_score,
                    "eval/total_score": total_score,
                }
            )
        wandb.finish()


if __name__ == "__main__":
    main()
