
"""
EVO AI - Training Engine
Version: 0.1.0

Trains the neural model from randomly initialized weights.
"""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from brain.model import EVOConfig, EVONeuralCore
from brain.tokenizer import EVOTokenizer
from training.dataset import EVODataset


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    config,
    output_path
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "config": config.__dict__
        },
        output_path
    )


def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"EVO AI device: {device}")

    tokenizer = EVOTokenizer()

    config = EVOConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.block_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )

    dataset = EVODataset(
        file_path=args.data,
        tokenizer=tokenizer,
        block_size=args.block_size
    )

    if len(dataset) < 2:
        raise ValueError(
            "Dataset is too small. Add more training text."
        )

    validation_size = max(
        1,
        int(len(dataset) * args.validation_split)
    )

    training_size = len(dataset) - validation_size

    training_data, validation_data = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size
    )

    model = EVONeuralCore(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate
    )

    print(
        f"Parameters: {model.parameter_count:,}"
    )
    print(
        f"Training examples: {len(training_data)}"
    )
    print(
        f"Validation examples: {len(validation_data)}"
    )

    for epoch in range(1, args.epochs + 1):

        model.train()
        total_train_loss = 0.0

        for input_ids, targets in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            _, loss = model(
                input_ids,
                targets
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        model.eval()
        total_validation_loss = 0.0

        with torch.no_grad():
            for input_ids, targets in validation_loader:
                input_ids = input_ids.to(device)
                targets = targets.to(device)

                _, loss = model(
                    input_ids,
                    targets
                )

                total_validation_loss += loss.item()

        average_validation_loss = (
            total_validation_loss / len(validation_loader)
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {average_train_loss:.4f} | "
            f"Validation loss: {average_validation_loss:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=average_validation_loss,
            config=config,
            output_path=args.output
        )

    print("Training completed.")
    print(f"Checkpoint saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Train EVO AI from scratch"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/evo_v0.1.pt"
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    if not 0 < args.validation_split < 1:
        parser.error("validation-split must be between 0 and 1")

    if args.block_size < 1:
        parser.error("block-size must be positive")

    train(args)


if __name__ == "__main__":
    main()
