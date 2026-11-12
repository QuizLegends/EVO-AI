
"""
EVO AI - Self Evaluation
Version: 0.1.0

Tracks model performance and detects improvements or regressions.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path


class EVOSelfEvaluator:

    def __init__(
        self,
        history_path="evolution/evaluation_history.json"
    ):
        self.history_path = Path(history_path)

        self.history_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not self.history_path.exists():
            self.history_path.write_text(
                "[]",
                encoding="utf-8"
            )

    @staticmethod
    def _validate_loss(value, name):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")

    def compare(
        self,
        previous_loss,
        current_loss,
        training_loss=None
    ):
        self._validate_loss(
            previous_loss,
            "previous_loss"
        )

        self._validate_loss(
            current_loss,
            "current_loss"
        )

        if training_loss is not None:
            self._validate_loss(
                training_loss,
                "training_loss"
            )

        if current_loss < previous_loss:
            status = "improved"

        elif current_loss > previous_loss:
            status = "regressed"

        else:
            status = "unchanged"

        if previous_loss > 0:
            improvement_percent = (
                (previous_loss - current_loss)
                / previous_loss
            ) * 100
        else:
            improvement_percent = 0.0

        overfitting_warning = False

        if training_loss is not None:
            overfitting_warning = (
                current_loss > training_loss * 1.5
                and current_loss - training_loss > 0.1
            )

        report = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "previous_validation_loss": previous_loss,
            "current_validation_loss": current_loss,
            "training_loss": training_loss,
            "improvement_percent": round(
                improvement_percent,
                4
            ),
            "status": status,
            "possible_overfitting": overfitting_warning
        }

        self._save_report(report)

        return report

    def _save_report(self, report):
        history = self.get_history()

        history.append(report)

        self.history_path.write_text(
            json.dumps(
                history,
                indent=4,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    def get_history(self):
        try:
            content = self.history_path.read_text(
                encoding="utf-8"
            )

            history = json.loads(content)

            if not isinstance(history, list):
                return []

            return history

        except (json.JSONDecodeError, OSError):
            return []

    def get_latest(self):
        history = self.get_history()

        if not history:
            return None

        return history[-1]
