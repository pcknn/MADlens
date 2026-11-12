"""Compare exported fixed predictions without loading or retraining any models."""

import argparse
from pathlib import Path

import pandas as pd

from evaluation import paired_comparison, validate_predictions, validation_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled", required=True, type=Path)
    parser.add_argument("--specialist", required=True, nargs="+", type=Path,
                        help="Disjoint files; duplicate paths across files are errors.")
    parser.add_argument("--validation-csv", type=Path,
                        default=Path("MURA-v1.1_files/valid_image_paths.csv"))
    parser.add_argument("--body-parts", nargs="+",
                        help="Explicit comparison subset, e.g. FOREARM WRIST. Default: all validation images.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pooled = pd.read_csv(args.pooled)
    specialist = pd.concat([pd.read_csv(p) for p in sorted(args.specialist)], ignore_index=True)
    # Check before filtering so overlapping specialist groups cannot hide duplicates.
    validate_predictions(pooled)
    validate_predictions(specialist)
    expected = validation_manifest(args.validation_csv)
    if args.body_parts:
        unknown = set(args.body_parts) - set(expected["body_part"])
        if unknown:
            raise ValueError(f"Unknown validation body parts: {sorted(unknown)}")
        print(f"Explicit comparison subset: {', '.join(args.body_parts)}")
        # Only the declared anatomy subset is excluded, never unmatched rows.
        pooled = pooled[pooled["body_part"].isin(args.body_parts)]
        specialist = specialist[specialist["body_part"].isin(args.body_parts)]
        expected = expected[expected["body_part"].isin(args.body_parts)]

    # Global validation happens before any overall/per-anatomy computations.
    overall = paired_comparison(pooled, specialist, expected)
    overall.insert(0, "body_part", "ALL_SELECTED")
    reports = [overall]
    for anatomy in sorted(expected["body_part"].unique()):
        report = paired_comparison(
            pooled[pooled["body_part"] == anatomy],
            specialist[specialist["body_part"] == anatomy],
            expected[expected["body_part"] == anatomy],
        )
        report.insert(0, "body_part", anatomy)
        reports.append(report)
    result = pd.concat(reports, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved paired patient bootstrap comparisons to {args.output}")


if __name__ == "__main__":
    main()
