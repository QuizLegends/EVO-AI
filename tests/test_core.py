
"""
EVO AI - Core Tests
Version: 0.1.0

Tests tokenizer, neural core, dataset and self-evaluation.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from brain.model import EVOConfig, EVONeuralCore
from brain.tokenizer import EVOTokenizer
from training.dataset import EVODataset
from evolution.self_evaluation import EVOSelfEvaluator


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = EVOTokenizer()

    def test_encode_decode_roundtrip(self):
        text = "EVO AI é inteligente!"
        tokens = self.tokenizer.encode(text)

        self.assertEqual(
            self.tokenizer.decode(tokens),
            text
        )

    def test_vocabulary_size(self):
        self.assertEqual(
            self.tokenizer.vocab_size,
            256
        )

    def test_invalid_token(self):
        with self.assertRaises(ValueError):
            self.tokenizer.decode([256])


class TestNeuralCore(unittest.TestCase):

    def setUp(self):
        self.config = EVOConfig(
            vocab_size=256,
            max_seq_len=16,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dropout=0.0
        )

        self.model = EVONeuralCore(self.config)

    def test_output_shape(self):
        inputs = torch.randint(0, 256, (2, 8))

        logits, loss = self.model(inputs)

        self.assertEqual(
            logits.shape,
            (2, 8, 256)
        )

        self.assertIsNone(loss)

    def test_training_loss(self):
        inputs = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))

        logits, loss = self.model(inputs, targets)

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())

    def test_generation(self):
        inputs = torch.randint(0, 256, (1, 5))

        generated = self.model.generate(
            inputs,
            max_new_tokens=3
        )

        self.assertEqual(
            generated.shape,
            (1, 8)
        )

    def test_sequence_limit(self):
        inputs = torch.randint(0, 256, (1, 17))

        with self.assertRaises(ValueError):
            self.model(inputs)


class TestDataset(unittest.TestCase):

    def test_shifted_sequences(self):

        with tempfile.TemporaryDirectory() as directory:

            path = Path(directory) / "data.txt"

            path.write_text(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                encoding="utf-8"
            )

            tokenizer = EVOTokenizer()

            dataset = EVODataset(
                path,
                tokenizer,
                block_size=4
            )

            inputs, targets = dataset[0]

            self.assertEqual(
                inputs.tolist(),
                [65, 66, 67, 68]
            )

            self.assertEqual(
                targets.tolist(),
                [66, 67, 68, 69]
            )


class TestSelfEvaluation(unittest.TestCase):

    def test_detects_improvement(self):

        with tempfile.TemporaryDirectory() as directory:

            history_path = (
                Path(directory) / "history.json"
            )

            evaluator = EVOSelfEvaluator(
                history_path=history_path
            )

            report = evaluator.compare(
                previous_loss=2.0,
                current_loss=1.5,
                training_loss=1.4
            )

            self.assertEqual(
                report["status"],
                "improved"
            )

            self.assertEqual(
                report["improvement_percent"],
                25.0
            )

            self.assertEqual(
                len(evaluator.get_history()),
                1
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
