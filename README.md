
# EVO AI

**EVO AI** is an experimental neural language model developed from scratch using PyTorch.

The project aims to build a programming-focused artificial intelligence with its own neural architecture, training pipeline, persistent memory and controlled improvement process.

## Project principles

- Neural network initialized with random weights.
- No pretrained model weights.
- Byte-level UTF-8 tokenizer.
- Transformer-based architecture.
- Training and evaluation with PyTorch.
- Persistent external memory.
- Evaluation-driven improvement proposals.
- Controlled experimentation and model versioning.

## Architecture

```text
EVO-AI/
├── brain/
│   ├── model.py
│   └── tokenizer.py
├── training/
│   ├── dataset.py
│   ├── train.py
│   └── evaluation.py
├── memory/
│   └── memory.py
├── evolution/
│   ├── self_evaluation.py
│   └── improvement.py
├── checkpoints/
├── notebooks/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer.
- PyTorch 2.2 or newer.
- CPU or compatible CUDA GPU.

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Preparing training data

Create a text file containing training material:

```text
datasets/train.txt
```

The dataset can contain programming examples, explanations, documentation and other relevant text.

Use only material that you have permission to use.

The dataset must contain more bytes than the configured block size.

## Training

From the repository root, execute:

```bash
python -m training.train --data datasets/train.txt
```

The default configuration is intentionally small for initial experiments.

## Evaluation

After training, evaluate the checkpoint:

```bash
python -m training.evaluation --data datasets/validation.txt --checkpoint checkpoints/evo_v0.1.pt
```

Use validation data that was not used for training.

## Memory

The `memory` module stores interactions and can export selected records as training text.

Stored interactions are not automatically learned by the neural network.

## Evolution

The `evolution` module analyzes evaluation history and generates improvement recommendations.

It does not automatically modify model weights.

## Current development status

EVO AI is under active development.

The current codebase provides an initial neural architecture and supporting infrastructure.

Training, integration tests, data quality checks, inference interfaces and controlled model improvement still need to be implemented and validated.

## Safety

Code execution and experimental training should take place in isolated environments with appropriate resource limits and permissions.

Never execute untrusted code directly on a personal device or production system.

## License

To be defined.
```

### Uma observação sobre a estrutura

O README já documenta a pasta `datasets`, que ainda precisaremos criar. Não precisa criá-la agora.

Quando terminar de enviar o arquivo, diga **Arquivo 10 concluído**.
