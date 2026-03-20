"""
Simulates a batch of new incoming data by sampling rows from the held-out
raw dataset, then triggers a retrain.

Usage:
  python simulate_batch.py              # sample 200 rows (default)
  python simulate_batch.py --n 500
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from dataset import RAW_PATH

PIPELINE_DIR = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of rows to sample as the new batch")
    args = parser.parse_args()

    df = pd.read_csv(RAW_PATH)
    batch = df.sample(n=min(args.n, len(df)), random_state=None)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        batch.to_csv(f.name, index=False)
        print(f"Sampled {len(batch)} rows → {f.name}")
        subprocess.run(
            ["python", "retrain.py", "--new-data", f.name],
            cwd=PIPELINE_DIR,
            check=True,
        )


if __name__ == "__main__":
    main()
