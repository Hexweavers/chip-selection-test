import csv
import json
from pathlib import Path

from models.chip import TestResult
from config import RESULTS_DIR, CHIP_TYPES


class ResultStorage:
    def __init__(self):
        self.results_dir = Path(RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True)

    def _get_model_dir(self, model: str) -> Path:
        # Convert model ID to safe directory name
        safe_name = model.replace("/", "--")
        model_dir = self.results_dir / safe_name
        model_dir.mkdir(exist_ok=True)
        return model_dir

    def _get_filename(self, result: TestResult) -> str:
        m = result.metadata
        return f"{m.persona_id}_{m.style}_{m.constraint}_{m.input_type}_{m.chip_count}"

    def save_result(self, result: TestResult) -> tuple[Path, Path]:
        """Save result as JSON and CSV. Returns (json_path, csv_path)."""
        model_dir = self._get_model_dir(result.metadata.model)
        filename = self._get_filename(result)

        # Save JSON
        json_path = model_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            f.write(result.to_json())

        # Save individual CSV
        csv_path = model_dir / f"{filename}.csv"
        self._write_single_csv(result, csv_path)

        # Append to summary CSV
        self._append_to_summary(result)

        return json_path, csv_path

    def _write_single_csv(self, result: TestResult, path: Path):
        """Write a single result to its own CSV file."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "display", "type", "source"])

            for chip in result.step1_chips:
                writer.writerow([chip.key, chip.display, chip.type, "step1"])
            for chip in result.user_selected_chips:
                writer.writerow([chip.key, chip.display, chip.type, "selected"])
            for chip in result.step2_chips:
                writer.writerow([chip.key, chip.display, chip.type, "step2"])
            for chip in result.fill_chips:
                writer.writerow([chip.key, chip.display, chip.type, "fill"])

    def _append_to_summary(self, result: TestResult):
        """Append result to the summary CSV."""
        summary_path = self.results_dir / "summary.csv"
        file_exists = summary_path.exists()

        m = result.metadata
        counts = result.count_by_type()

        row = {
            "model": m.model,
            "persona_id": m.persona_id,
            "sector": m.sector,
            "desired_role": m.desired_role,
            "style": m.style,
            "constraint": m.constraint,
            "input_type": m.input_type,
            "chip_count_requested": m.chip_count,
            "final_chip_count": len(result.final_chips),
            "situation_count": counts.get("situation", 0),
            "jargon_count": counts.get("jargon", 0),
            "role_task_count": counts.get("role_task", 0),
            "environment_count": counts.get("environment", 0),
            "fill_needed": len(result.fill_chips) > 0,
            "fill_count": len(result.fill_chips),
            "step1_chips_json": json.dumps([c.to_dict() for c in result.step1_chips]),
            "selected_chips_json": json.dumps(
                [c.to_dict() for c in result.user_selected_chips]
            ),
            "final_chips_json": json.dumps([c.to_dict() for c in result.final_chips]),
            "errors": "; ".join(result.errors) if result.errors else "",
            "timestamp": m.timestamp,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def result_exists(
        self,
        model: str,
        persona_id: str,
        style: str,
        constraint: str,
        input_type: str,
        chip_count: int,
    ) -> bool:
        """Check if a result already exists (for resume functionality)."""
        model_dir = self._get_model_dir(model)
        filename = f"{persona_id}_{style}_{constraint}_{input_type}_{chip_count}.json"
        return (model_dir / filename).exists()

    def clear_summary(self):
        """Clear the summary CSV (useful for fresh runs)."""
        summary_path = self.results_dir / "summary.csv"
        if summary_path.exists():
            summary_path.unlink()
