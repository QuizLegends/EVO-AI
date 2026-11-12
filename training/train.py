"""
EVO AI - Training Engine
Version: 0.2.0

Trains the neural model from randomly initialized weights.

Features:
- Independent programming examples.
- Variable-length batch support.
- Padding ignored during loss calculation.
- Gradient accumulation.
- Optional CUDA mixed precision.
- Token-weighted validation loss.
- Reproducible train/validation split.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

from brain.model import EVOConfig, EVONeuralCore
from brain.tokenizer import EVOTokenizer
from training.dataset import EVODataset


IGNORE_INDEX = -100


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_examples(batch):
    """
    Pads variable-length examples.

    Padding targets use IGNORE_INDEX so they do not
    contribute to the training loss.
    """

    max_length = max(
        len(input_ids)
        for input_ids, _ in batch
    )

    batch_size = len(batch)

    input_ids_batch = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long
    )

    targets_batch = torch.full(
        (batch_size, max_length),
        IGNORE_INDEX,
        dtype=torch.long
    )

    for index, (input_ids, targets) in enumerate(batch):

        length = len(input_ids)

        input_ids_batch[index, :length] = input_ids
        targets_batch[index, :length] = targets

    return input_ids_batch, targets_batch


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


def evaluate(
    model,
    loader,
    device,
    use_amp
):
    """
    Calculates validation loss weighted by valid tokens.
    """

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():

        for input_ids, targets in loader:

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            valid_tokens = (
                targets != IGNORE_INDEX
            ).sum().item()

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                logits, _ = model(
                    input_ids,
                    targets
                )

                loss_sum = F.cross_entropy(
                    logits.reshape(
                        -1,
                        model.config.vocab_size
                    ),
                    targets.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum"
                )

            total_loss += loss_sum.item()
            total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError(
            "Validation dataset contains no valid tokens."
        )

    return total_loss / total_tokens


def train(args):

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and not args.no_mixed_precision
    )

    print("EVO AI device:", device)
    print("Mixed precision:", use_amp)

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
            "Dataset needs at least two examples."
        )

    validation_size = max(
        1,
        int(len(dataset) * args.validation_split)
    )

    training_size = (
        len(dataset) - validation_size
    )

    if training_size < 1:
        raise ValueError(
            "Training dataset is too small."
        )

    training_data, validation_data = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_examples,
        pin_memory=(device.type == "cuda")
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_examples,
        pin_memory=(device.type == "cuda")
    )

    model = EVONeuralCore(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    print("Parameters:", f"{model.parameter_count:,}")
    print("Total examples:", len(dataset))
    print("Training examples:", len(training_data))
    print("Validation examples:", len(validation_data))
    print("Batch size:", args.batch_size)
    print(
        "Gradient accumulation:",
        args.gradient_accumulation_steps
    )
    print("Context:", args.block_size)

    for epoch in range(1, args.epochs + 1):

        model.train()

        optimizer.zero_grad(set_to_none=True)

        total_train_loss = 0.0
        total_train_tokens = 0

        for step, (input_ids, targets) in enumerate(
            train_loader
        ):

            input_ids = input_ids.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            valid_tokens = (
                targets != IGNORE_INDEX
            ).sum().item()

            if valid_tokens == 0:
                continue

            group_start = (
                step // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps

            group_size = min(
                args.gradient_accumulation_steps,
                len(train_loader) - group_start
            )

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                _, loss = model(
                    input_ids,
                    targets
                )

                scaled_loss = loss / group_size

            scaler.scale(scaled_loss).backward()

            total_train_loss += (
                loss.detach().item() * valid_tokens
            )

            total_train_tokens += valid_tokens

            is_accumulation_step = (
                (step + 1)
                % args.gradient_accumulation_steps == 0
            )

            is_last_step = (
                step + 1 == len(train_loader)
            )

            if is_accumulation_step or is_last_step:

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

        if total_train_tokens == 0:
            raise ValueError(
                "Training dataset contains no valid tokens."
            )

        average_train_loss = (
            total_train_loss / total_train_tokens
        )

        average_validation_loss = evaluate(
            model,
            validation_loader,
            device,
            use_amp
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
    print("Checkpoint saved to:", args.output)


def main():

    parser = argparse.ArgumentParser(
        description="Train EVO AI from scratch."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/evo_v0.2.pt"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=8192
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4
    )

    parser.add_argument(
        "--no-mixed-precision",
        action="store_true"
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=128
    )

    parser.add_argument(
        "--n-heads",
        type=int,
        default=4
    )

    parser.add_argument(
        "--n-layers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("epochs must be positive")

    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    if args.gradient_accumulation_steps < 1:
        parser.error(
            "gradient-accumulation-steps must be positive"
        )

    if not 0 < args.validation_split < 1:
        parser.error(
            "validation-split must be between 0 and 1"
        )

    if args.block_size < 2:
        parser.error("block-size must be at least 2")

    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")

    train(args)


if __name__ == "__main__":
    main()
