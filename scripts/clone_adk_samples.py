from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/adk_samples")
    args = parser.parse_args()
    dest = Path(args.dest)
    if dest.exists():
        print(f"already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/google/adk-samples.git", str(dest)],
        check=True,
    )
    print(f"cloned: {dest}")


if __name__ == "__main__":
    main()
