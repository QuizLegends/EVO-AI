
"""
EVO AI - Model Evaluation
Version: 0.1.0

Evaluates a trained model using validation data.
"""

import argparse
import math

import torch
from torch.utils.data import DataLoader

from brain.model import EVOConfig, EVONeuralCore
from brain.tokenizer import EVOTokenizer
from training.dataset import EVODataset


def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Evaluation device: {device}")

    # Load checkpoint from a trusted source.
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True
    )

    config = EVOConfig(**checkpoint["config"])

    tokenizer = EVOTokenizer()

    dataset = EVODataset(
        file_path=args.data,
        tokenizer=tokenizer,
        block_size=config.max_seq_len
    )

    if len(dataset) == 0:
        raise ValueError("Not enough data for evaluation.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = EVONeuralCore(config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            _, loss = model(
                input_ids,
                targets
            )

            token_count = targets.numel()

            total_loss += loss.item() * token_count
            total_tokens += token_count

    average_loss = total_loss / total_tokens

    perplexity = math.exp(
        min(average_loss, 20)
    )

    print("\n===== EVO AI EVALUATION =====")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    print(f"Evaluation loss: {average_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Evaluated tokens: {total_tokens:,}")
    print("=============================")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an EVO AI checkpoint"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    evaluate(args)


if __name__ == "__main__":
    main()
