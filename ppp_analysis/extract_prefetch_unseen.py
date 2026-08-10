#!/usr/bin/env python3
"""Extract prefetch_unseen_frac/0 from TensorBoard event files.

The script reads the eight final-run vis_data directories used by option 6 of
jsc_jupiter/tensorboard.sh. It writes one CSV per model and one summary CSV.

No TensorBoard server is started and no event file is modified.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util


TAG = "prefetch_unseen_frac/0"

LOG_DIRS: dict[str, str] = {
    "UNet": (
        "/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/"
        "1041277_ppp_finals_unet_54k/20260725_062756/vis_data"
    ),
    "UNet_woVel": (
        "/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/"
        "1041278_ppp_finals_unet_54k_woVel/20260725_062804/vis_data"
    ),
    "UNet_woVel_woRCS": (
        "/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/"
        "1047407_ppp_finals_unet_54k_woVel_woRCS/"
        "20260726_145120/vis_data"
    ),
    "UperNet": (
        "/e/scratch/nxtaim-1/huber7/data/260725_finals/run_2/"
        "1096822_ppp_finals_upernet_run3_50k/"
        "20260730_015303/vis_data"
    ),
    "TrafoEnc_CamDom": (
        "/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/"
        "1041091_ppp_finals_TrafoEnc_CamDom_24k/"
        "20260725_060602/vis_data"
    ),
    "TrafoEnc_BEVpolDom": (
        "/e/scratch/nxtaim-1/huber7/data/260725_finals/run_1/"
        "1041092_ppp_finals_TrafoEnc_BEVpolDom_20k/"
        "20260725_060631/vis_data"
    ),
    "TrafoEncDec": (
        "/e/scratch/nxtaim-1/huber7/data/260725_finals/run_2/"
        "1065402_ppp_finals_TrafoEncDec_run2_20k/"
        "20260727_201415/vis_data"
    ),
    "TrafoEncDec_woVel": (
        "/e/scratch/nxtaim-1/huber7/work_dirs/"
        "1096858_ppp_finals_TrafoEncDec_run2_20k_woVel/"
        "20260730_015622/vis_data"
    ),
}


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    wall_time: float
    value: float
    event_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TensorBoard prefetch unseen fractions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "prefetch_unseen_results",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of runs processed concurrently (default: 2).",
    )
    parser.add_argument(
        "--tag",
        default=TAG,
        help=f"TensorBoard tag to extract (default: {TAG}).",
    )
    return parser.parse_args()


def event_files(log_dir: Path) -> list[Path]:
    """Return TensorBoard event files below a vis_data directory."""
    return sorted(
        path
        for path in log_dir.rglob("events.out.tfevents.*")
        if path.is_file()
    )


def scalar_from_summary_value(summary_value) -> float | None:
    """Read a scalar from either simple_value or tensor storage."""
    field = summary_value.WhichOneof("value")

    if field == "simple_value":
        return float(summary_value.simple_value)

    if field == "tensor":
        array = tensor_util.make_ndarray(summary_value.tensor)
        if array.size != 1:
            return None
        return float(array.reshape(-1)[0])

    return None


def read_tag_from_event_file(path: Path, tag: str) -> list[ScalarPoint]:
    points: list[ScalarPoint] = []
    loader = event_file_loader.EventFileLoader(str(path))

    for event in loader.Load():
        if not event.HasField("summary"):
            continue

        for summary_value in event.summary.value:
            if summary_value.tag != tag:
                continue

            value = scalar_from_summary_value(summary_value)
            if value is None or not math.isfinite(value):
                continue

            points.append(
                ScalarPoint(
                    step=int(event.step),
                    wall_time=float(event.wall_time),
                    value=value,
                    event_file=str(path),
                )
            )

    return points


def deduplicate(points: Iterable[ScalarPoint]) -> list[ScalarPoint]:
    """Keep the latest event for each step, handling restarted runs."""
    by_step: dict[int, ScalarPoint] = {}
    for point in points:
        previous = by_step.get(point.step)
        if previous is None or point.wall_time >= previous.wall_time:
            by_step[point.step] = point
    return [by_step[step] for step in sorted(by_step)]


def arithmetic_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def step_weighted_mean(points: list[ScalarPoint]) -> float:
    """Weight each logged rate by the step interval ending at that point."""
    if not points:
        return math.nan

    weighted_sum = 0.0
    total_weight = 0
    previous_step = 0

    for point in points:
        interval = point.step - previous_step
        if interval > 0:
            weighted_sum += interval * point.value
            total_weight += interval
        previous_step = point.step

    return weighted_sum / total_weight if total_weight else math.nan


def write_points_csv(path: Path, model: str, points: list[ScalarPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "step", "wall_time", "unseen_fraction", "event_file"]
        )
        for point in points:
            writer.writerow(
                [model, point.step, point.wall_time, point.value, point.event_file]
            )


def process_run(
    model: str,
    log_dir_string: str,
    tag: str,
    output_dir_string: str,
) -> dict[str, object]:
    log_dir = Path(log_dir_string)
    output_dir = Path(output_dir_string)

    if not log_dir.is_dir():
        return {
            "model": model,
            "status": "missing_directory",
            "message": str(log_dir),
        }

    files = event_files(log_dir)
    if not files:
        return {
            "model": model,
            "status": "no_event_files",
            "message": str(log_dir),
        }

    all_points: list[ScalarPoint] = []
    for path in files:
        all_points.extend(read_tag_from_event_file(path, tag))

    points = deduplicate(all_points)
    if not points:
        return {
            "model": model,
            "status": "tag_not_found",
            "message": f"{tag} not found in {len(files)} event file(s)",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_points_csv(
        output_dir / f"prefetch_unseen_{model}.csv",
        model,
        points,
    )

    values = [point.value for point in points]
    return {
        "model": model,
        "status": "ok",
        "log_dir": str(log_dir),
        "n_event_files": len(files),
        "n_logged_values": len(points),
        "first_step": points[0].step,
        "last_step": points[-1].step,
        "mean_unseen_fraction": arithmetic_mean(values),
        "step_weighted_unseen_fraction": step_weighted_mean(points),
        "final_unseen_fraction": points[-1].value,
        "min_unseen_fraction": min(values),
        "max_unseen_fraction": max(values),
        "message": "",
    }


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "model",
        "status",
        "log_dir",
        "n_event_files",
        "n_logged_values",
        "first_step",
        "last_step",
        "mean_unseen_fraction",
        "step_weighted_unseen_fraction",
        "final_unseen_fraction",
        "min_unseen_fraction",
        "max_unseen_fraction",
        "message",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def main() -> int:
    args = parse_args()
    workers = max(1, min(args.workers, len(LOG_DIRS)))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tag:        {args.tag}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Workers:    {workers}", flush=True)
    print(f"Runs:       {len(LOG_DIRS)}", flush=True)
    print(flush=True)

    rows: list[dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_run,
                model,
                log_dir,
                args.tag,
                str(output_dir),
            ): model
            for model, log_dir in LOG_DIRS.items()
        }

        for future in as_completed(futures):
            model = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # Keep other runs alive if one fails.
                row = {
                    "model": model,
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }

            rows.append(row)
            status = row.get("status")
            if status == "ok":
                weighted = float(row["step_weighted_unseen_fraction"])
                print(
                    f"[DONE]   {model:<24} "
                    f"weighted unseen fraction = {weighted:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[FAILED] {model:<24} {status}: {row.get('message', '')}",
                    flush=True,
                )

    rows.sort(key=lambda row: str(row.get("model", "")))
    summary_path = output_dir / "prefetch_unseen_summary.csv"
    write_summary_csv(summary_path, rows)

    print(flush=True)
    print(f"Summary written to: {summary_path}", flush=True)
    print("Per-model CSV files are in the same directory.", flush=True)

    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        print(f"Completed with {len(failed)} failed run(s).", flush=True)
        return 2

    print("All runs completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())