
"""
EVO AI - Training Integration Test
Version: 0.1.0

Verifies that a real training step updates model weights
and produces a valid checkpoint.
"""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from brain.model import EVOConfig, EVONeuralCore
from training.train import train, set_seed


class TestTrainingIntegration(unittest.TestCase):

    def test_training_updates_weights_and_saves_checkpoint(self):

        with tempfile.TemporaryDirectory() as directory:

            root = Path(directory)

            data_path = root / "training.txt"
            checkpoint_path = root / "evo_test.pt"

            data_path.write_text(
                (
                    "EVO AI learns programming and language. "
                    "Neural networks learn patterns from data. "
                    "Training updates model parameters. "
                ) * 10,
                encoding="utf-8"
            )

            seed = 42

            config = EVOConfig(
                vocab_size=256,
                max_seq_len=4,
                d_model=8,
                n_heads=2,
                n_layers=1,
                dropout=0.0
            )

            # Save the initial weights for comparison.
            set_seed(seed)

            initial_model = EVONeuralCore(config)

            initial_weights = {
                name: parameter.detach().clone()
                for name, parameter
                in initial_model.named_parameters()
            }

            del initial_model

            args = Namespace(
                seed=seed,
                data=str(data_path),
                output=str(checkpoint_path),
                epochs=1,
                batch_size=2,
                block_size=4,
                d_model=8,
                n_heads=2,
                n_layers=1,
                dropout=0.0,
                learning_rate=0.001,
                validation_split=0.2
            )

            # Execute actual training.
            train(args)

            self.assertTrue(
                checkpoint_path.is_file()
            )

            # Load the checkpoint.
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True
            )

            self.assertEqual(
                checkpoint["epoch"],
                1
            )

            trained_weights = checkpoint[
                "model_state_dict"
            ]

            # Confirm that at least one parameter changed.
            changed_parameters = [
                name
                for name, initial in initial_weights.items()
                if not torch.equal(
                    initial,
                    trained_weights[name]
                )
            ]

            self.assertGreater(
                len(changed_parameters),
                0,
                "Training did not update any model weights."
            )

            self.assertTrue(
                torch.isfinite(
                    torch.tensor(checkpoint["loss"])
                ).item()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
