import json
import subprocess
import sys


def test_promotion_cli_runs(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"coverage": 0.95, "rmse": 5.0, "drift_score": 0.01}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.promote_model",
            "model-1",
            "--metrics-file",
            str(metrics_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert '"approved": true' in result.stdout.lower()
