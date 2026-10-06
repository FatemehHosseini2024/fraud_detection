"""
Assemble creditcard.csv from the two git-tracked parts.

GitHub rejects files > 100 MB in a regular git push, so the dataset is
committed as two ~72 MB parts (creditcard_part1.csv + creditcard_part2.csv).
Run this script once after cloning to reconstruct the full CSV:

    python assemble_data.py

The pipeline (pipeline.py) also calls this automatically if creditcard.csv
is missing, so normal usage (python pipeline.py, streamlit run app.py, ...)
works without any manual step.
"""

import os

DATA_PATH = "creditcard.csv"
PARTS = ["creditcard_part1.csv", "creditcard_part2.csv"]


def assemble(output_path=DATA_PATH, parts=PARTS):
    """Concatenate the part files into the full dataset CSV."""
    if os.path.exists(output_path):
        print(f"{output_path} already exists, nothing to do.")
        return output_path

    missing = [p for p in parts if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing part file(s): {missing}. "
            "Download the dataset from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and save it as {output_path}, or re-download the part files."
        )

    print(f"Assembling {output_path} from {len(parts)} parts...")
    with open(output_path, "wb") as out:
        for part in parts:
            with open(part, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            print(f"  + {part}")

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Done. {output_path} is {size_mb:.1f} MB.")
    return output_path


if __name__ == "__main__":
    assemble()
