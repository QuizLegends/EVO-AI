
"""
EVO AI - Interactive Interface
Version: 0.1.0

Loads a trained checkpoint and provides terminal interaction.
"""

import argparse

import torch

from brain.model import EVOConfig, EVONeuralCore
from brain.tokenizer import EVOTokenizer
from memory.memory import EVOMemory


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    config = EVOConfig(**checkpoint["config"])

    model = EVONeuralCore(config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def generate_response(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
):
    tokens = tokenizer.encode(prompt)

    if not tokens:
        raise ValueError("Prompt cannot be empty.")

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    generated_tokens = output_ids[0].tolist()

    response = tokenizer.decode(
        generated_tokens[len(tokens):]
    )

    return response


def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"EVO AI device: {device}")
    print("Loading neural model...")

    model, checkpoint = load_model(
        args.checkpoint,
        device
    )

    tokenizer = EVOTokenizer()
    memory = EVOMemory(args.memory)

    print("Model loaded successfully.")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print("Type 'exit' to close the conversation.")
    print("-" * 45)

    while True:
        try:
            prompt = input("\nYou: ").strip()

            if prompt.lower() in ("exit", "quit", "sair"):
                print("EVO AI session closed.")
                break

            if not prompt:
                continue

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )

            print(f"\nEVO AI: {response}")

            memory.add(
                role="user",
                content=prompt,
                tags=["conversation"]
            )

            memory.add(
                role="assistant",
                content=response,
                tags=["generated_response"]
            )

        except KeyboardInterrupt:
            print("\nEVO AI session interrupted.")
            break

        except Exception as error:
            print(
                f"\nInteraction error: {error}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Interact with a trained EVO AI model."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--memory",
        type=str,
        default="memory/data.jsonl"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40
    )

    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")

    if args.temperature <= 0:
        parser.error("temperature must be positive")

    if args.top_k < 1:
        parser.error("top-k must be positive")

    run(args)


if __name__ == "__main__":
    main()
