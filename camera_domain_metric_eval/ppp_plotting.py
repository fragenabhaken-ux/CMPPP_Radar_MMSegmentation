"""Plotting helpers for camera-domain metric CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_expected_count_scatter(
    df_samples: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, plt.Figure]:
    required = {"model", "gt_count", "expected_count"}
    missing = required - set(df_samples.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    out = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    figures: dict[str, plt.Figure] = {}
    for model, group in df_samples.groupby("model", dropna=False):
        x = pd.to_numeric(group["gt_count"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group["expected_count"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, alpha=0.5, s=20)
        if x.size:
            lower = float(min(x.min(), y.min()))
            upper = float(max(x.max(), y.max()))
            ax.plot([lower, upper], [lower, upper], linestyle="--")
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Ground-truth count")
        ax.set_ylabel("Expected count")
        ax.set_title(f"Expected count vs. ground-truth count\n{model}")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        model_key = str(model)
        figures[model_key] = fig
        if out is not None:
            safe = model_key.replace("/", "_").replace("\\", "_")
            fig.savefig(out / f"{safe}_expected_count_scatter.png", dpi=300, bbox_inches="tight")
    return figures


def plot_metric_boxplots(
    df_samples: pd.DataFrame,
    metrics: Iterable[str],
    *,
    output_dir: str | Path | None = None,
    showfliers: bool = False,
) -> dict[str, plt.Figure]:
    out = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    figures: dict[str, plt.Figure] = {}
    for metric in metrics:
        if metric not in df_samples:
            raise KeyError(f"Column not found: {metric}")
        fig, ax = plt.subplots(figsize=(8, 5))
        df_samples.boxplot(column=metric, by="model", ax=ax, showfliers=showfliers)
        fig.suptitle("")
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        figures[metric] = fig
        if out is not None:
            fig.savefig(out / f"boxplot_{metric}.png", dpi=300, bbox_inches="tight")
    return figures


def plot_metric_vs_gt_count(
    df_samples: pd.DataFrame,
    metric: str,
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    if metric not in df_samples:
        raise KeyError(f"Column not found: {metric}")
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, group in df_samples.groupby("model", dropna=False):
        ax.scatter(group["gt_count"], group[metric], alpha=0.45, s=18, label=model)
    ax.set_xlabel("Ground-truth count")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs. ground-truth count")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig
