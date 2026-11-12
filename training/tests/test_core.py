
"""
EVO AI - Core Tests
Version: 0.1.0

Automated tests for the initial EVO AI components.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from brain.model import EVOConfig, EVONeuralCore
from brain.tokenizer import EVOTokenizer
from training.dataset import EVODataset
from memory.memory import EVOMemory
from evolution.self_evaluation import EVOSelfEvaluator


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = EVOTokenizer()

    def test_encode_decode(self):
        text = "EVO AI - Olá! 🚀"

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


class TestNeuralModel(unittest.TestCase):

    def setUp(self):
        self.config = EVOConfig(
            vocab_size=256,
            max_seq_len=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            dropout=0.0
        )

        self.model = EVONeuralCore(self.config)

    def test_forward_output_shape(self):
        inputs = torch.randint(0, 256, (2, 4))

        logits, loss = self.model(inputs)

        self.assertEqual(
            tuple(logits.shape),
            (2, 4, 256)
        )

        self.assertIsNone(loss)

    def test_training_loss(self):
        inputs = torch.randint(0, 256, (2, 4))
        targets = torch.randint(0, 256, (2, 4))

        _, loss = self.model(inputs, targets)

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())

    def test_generation(self):
        inputs = torch.randint(0, 256, (1, 3))

        generated = self.model.generate(
            inputs,
            max_new_tokens=2
        )

        self.assertEqual(
            generated.shape[1],
            5
        )

    def test_sequence_limit(self):
        inputs = torch.randint(0, 256, (1, 9))

        with self.assertRaises(ValueError):
            self.model(inputs)


class TestDataset(unittest.TestCase):

    def test_dataset_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.txt"

            data_path.write_text(
                "abcdefghijklmnop",
                encoding="utf-8"
            )

            dataset = EVODataset(
                file_path=data_path,
                tokenizer=EVOTokenizer(),
                block_size=15
            )

            self.assertEqual(len(dataset), 1)

            inputs, targets = dataset[0]

            self.assertEqual(inputs.shape, (15,))
            self.assertEqual(targets.shape, (15,))

            self.assertEqual(
                inputs[1:].tolist(),
                targets[:-1].tolist()
            )


class TestMemory(unittest.TestCase):

    def test_save_and_retrieve(self):
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "memory.jsonl"

            memory = EVOMemory(memory_path)

            memory.add(
                "user",
                "Test EVO AI",
                ["test"]
            )

            memory.add(
                "assistant",
                "Test response",
                ["test"]
            )

            self.assertEqual(memory.count(), 2)

            results = memory.search("EVO AI")

            self.assertEqual(len(results), 1)
            self.assertEqual(
                results[0]["role"],
                "user"
            )


class TestSelfEvaluation(unittest.TestCase):

    def test_improvement_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = (
                Path(directory) / "history.json"
            )

            evaluator = EVOSelfEvaluator(history_path)

            report = evaluator.compare(
                previous_loss=2.0,
                current_loss=1.5,
                training_loss=1.3
            )

            self.assertEqual(
                report["status"],
                "improved"
            )

            self.assertEqual(
                report["improvement_percent"],
                25.0
            )

            self.assertTrue(history_path.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
