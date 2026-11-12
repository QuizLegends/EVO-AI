
"""
EVO AI - Improvement Planner
Version: 0.1.0

Analyzes evaluation history and proposes training improvements.
Does not modify model weights or training configurations automatically.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class EVOImprovementPlanner:

    def __init__(
        self,
        history_path="evolution/evaluation_history.json",
        report_path="evolution/improvement_report.json"
    ):
        self.history_path = Path(history_path)
        self.report_path = Path(report_path)

    def _load_history(self):
        if not self.history_path.is_file():
            return []

        try:
            history = json.loads(
                self.history_path.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(history, list):
                return [
                    item for item in history
                    if isinstance(item, dict)
                ]

        except (json.JSONDecodeError, OSError):
            pass

        return []

    def analyze(self):
        history = self._load_history()

        recommendations = []
        evidence = []

        if not history:
            recommendations.append(
                "Execute the first training and evaluation "
                "before proposing model improvements."
            )

            return self._create_report(
                "insufficient_data",
                recommendations,
                evidence
            )

        latest = history[-1]

        current_loss = latest.get(
            "current_validation_loss"
        )

        training_loss = latest.get(
            "training_loss"
        )

        if len(history) < 2:
            recommendations.append(
                "Collect at least two evaluation results "
                "to identify a performance trend."
            )

        else:
            previous = history[-2]

            previous_loss = previous.get(
                "current_validation_loss"
            )

            if (
                isinstance(current_loss, (int, float))
                and isinstance(previous_loss, (int, float))
            ):
                if current_loss > previous_loss:
                    evidence.append(
                        "Validation loss increased."
                    )

                    recommendations.append(
                        "Review the dataset, learning rate "
                        "and training configuration."
                    )

                elif current_loss < previous_loss:
                    evidence.append(
                        "Validation loss decreased."
                    )

                    recommendations.append(
                        "Preserve the current configuration "
                        "and verify whether the improvement "
                        "continues in later evaluations."
                    )

                else:
                    evidence.append(
                        "Validation loss remained unchanged."
                    )

                    recommendations.append(
                        "Investigate whether training is "
                        "making sufficient progress."
                    )

        if latest.get("possible_overfitting") is True:
            evidence.append(
                "Possible overfitting was detected."
            )

            recommendations.append(
                "Increase the diversity of training data "
                "and review regularization."
            )

        if (
            isinstance(current_loss, (int, float))
            and isinstance(training_loss, (int, float))
            and current_loss > training_loss
        ):
            evidence.append(
                "Validation loss is higher than training loss."
            )

            recommendations.append(
                "Check whether training and validation "
                "datasets represent similar content."
            )

        if not recommendations:
            recommendations.append(
                "Collect more evaluation data before "
                "changing the training configuration."
            )

        return self._create_report(
            "analysis_completed",
            recommendations,
            evidence
        )

    def _create_report(
        self,
        status,
        recommendations,
        evidence
    ):
        report = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "status": status,

            "evidence": evidence,

            "recommendations": recommendations,

            "automatic_changes_applied": False
        }

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.report_path.write_text(
            json.dumps(
                report,
                indent=4,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        return report


if __name__ == "__main__":
    planner = EVOImprovementPlanner()

    result = planner.analyze()

    print("\n===== EVO AI IMPROVEMENT REPORT =====")
    print(json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    ))
    print("=====================================")
