
"""
EVO AI - Neural Core
Version: 0.2.0

Transformer language model initialized from scratch.
No pretrained weights are loaded.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100


@dataclass
class EVOConfig:
    vocab_size: int = 256
    max_seq_len: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        if self.vocab_size < 2:
            raise ValueError("Invalid vocabulary size")

        if self.max_seq_len < 1:
            raise ValueError("Invalid sequence length")


class EVOBlock(nn.Module):
    """One causal Transformer block."""

    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(config.d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout)
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, padding_mask=None):
        seq_len = x.size(1)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        normalized = self.norm1(x)

        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False
        )

        x = x + self.dropout(attended)

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


class EVONeuralCore(nn.Module):
    """
    EVO AI's initial neural language model.

    Inputs: token IDs [batch, sequence]
    Outputs: vocabulary logits and optional training loss.
    """

    def __init__(self, config=None):
        super().__init__()

        self.config = config or EVOConfig()

        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.d_model
        )

        self.position_embedding = nn.Embedding(
            self.config.max_seq_len,
            self.config.d_model
        )

        self.blocks = nn.ModuleList([
            EVOBlock(self.config)
            for _ in range(self.config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(
            self.config.d_model
        )

        self.output_head = nn.Linear(
            self.config.d_model,
            self.config.vocab_size,
            bias=False
        )

        self.apply(self._initialize_weights)

        self.parameter_count = sum(
            p.numel() for p in self.parameters()
        )

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                "Sequence exceeds max_seq_len"
            )

        padding_mask = input_ids.eq(0)

        # The first position must always remain available
        # to prevent attention rows with every key masked.
        padding_mask[:, 0] = False

        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(
                x,
                padding_mask=padding_mask
            )

        x = self.final_norm(x)

        logits = self.output_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    self.config.vocab_size
                ),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None
    ):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        was_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):

                context = input_ids[
                    :, -self.config.max_seq_len:
                ]

                logits, _ = self(context)

                logits = logits[:, -1, :]
                logits = logits / temperature

                if top_k is not None:
                    k = min(top_k, logits.size(-1))

                    if k < 1:
                        raise ValueError("top_k must be positive")

                    values, _ = torch.topk(logits, k)

                    threshold = values[:, -1].unsqueeze(-1)

                    logits = logits.masked_fill(
                        logits < threshold,
                        float("-inf")
                    )

                probabilities = F.softmax(
                    logits,
                    dim=-1
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1
                )

                input_ids = torch.cat(
                    [input_ids, next_token],
                    dim=1
                )

            return input_ids

        finally:
            self.train(was_training)
