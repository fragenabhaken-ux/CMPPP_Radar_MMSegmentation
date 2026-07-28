"""Quick structural and count-scaling checks for exported camera HDF5 files."""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from ppp_metric_io_tools import list_h5_samples_for_model, load_h5_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_dir")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    entries = list_h5_samples_for_model("model", args.h5_dir)
    if args.max_samples is not None:
        entries = entries[: args.max_samples]

    uids: list[str] = []
    expected: list[float] = []
    gt_counts: list[int] = []
    shapes: Counter[tuple[int, int]] = Counter()

    for entry in entries:
        sample = load_h5_sample(entry)
        uids.append(sample["meta"]["sample_uid"])
        expected.append(float(np.asarray(sample["pred_lambda"]).sum()))
        gt_counts.append(int((np.asarray(sample["gt_support"]) > 0).sum()))
        shapes[(sample["shape"]["H"], sample["shape"]["W"])] += 1

    duplicates = len(uids) - len(set(uids))
    print(f"samples: {len(entries)}")
    print(f"duplicate sample_uid values: {duplicates}")
    print(f"shapes: {dict(shapes)}")
    print(
        "expected count: "
        f"mean={np.mean(expected):.3f}, min={np.min(expected):.3f}, max={np.max(expected):.3f}"
    )
    print(
        "GT count: "
        f"mean={np.mean(gt_counts):.3f}, min={np.min(gt_counts)}, max={np.max(gt_counts)}"
    )


if __name__ == "__main__":
    main()
