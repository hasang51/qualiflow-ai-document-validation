"""Offline M2 gold evaluation (mock predictions, no live Bedrock)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_outputs import run_academic_evaluation

DEFAULT_METADATA = PROJECT_ROOT / "data" / "gold" / "m2" / "metadata.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data" / "gold" / "m2" / "predictions"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "eval_runs" / "m2_offline"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic M2 gold evaluation.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = run_academic_evaluation(
        metadata_path=args.metadata,
        predictions_dir=args.predictions,
        out_dir=args.out_dir,
    )
    print(summary)
    if float(summary.get("unsafe_auto_accept_rate") or 0) != 0.0:
        print("FAIL: unsafe_auto_accept_rate must be 0", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
