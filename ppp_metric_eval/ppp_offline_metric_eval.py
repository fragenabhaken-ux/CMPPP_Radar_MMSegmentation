"""
Offline runner for PPP model comparison on exported HDF5 metric data.

This file is intentionally only the orchestration layer.

Responsibilities:
    - load metric_eval_config.py
    - iterate over model H5 exports
    - load samples via ppp_metric_io_tools.py
    - compute enabled metrics via metric-specific tools
    - aggregate sample-level metrics to model-level summary
    - write CSV outputs

For now this runner supports MMD-only evaluation.

Expected local files in the same folder:
    metric_eval_config.py
    ppp_metric_io_tools.py
    ppp_mmd_tools.py
    ppp_metrics_loss_mmd_tools.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from ppp_metric_io_tools import list_h5_samples_for_model, load_h5_sample
from ppp_mmd_tools import compute_mmd_metrics
from ppp_expected_count_tools import compute_expected_count_metrics
from ppp_depth_tools import compute_depth_metrics


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_python_config(config_path: str | os.PathLike[str]) -> ModuleType:
    """
    Load a Python config file as a module.

    Example:
        cfg = load_python_config("metric_eval_config.py")
        print(cfg.H5_DIRS)
    """
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("metric_eval_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _get_cfg_attr(cfg: ModuleType, name: str, default: Any = None) -> Any:
    """Small helper to read optional config attributes."""
    return getattr(cfg, name, default)


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(
    sample: dict[str, Any],
    *,
    metrics_cfg: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute all enabled metrics for one loaded H5 sample.
    """
    row: dict[str, Any] = {}

    expected_count_cfg = metrics_cfg.get("expected_count", {})
    if bool(expected_count_cfg.get("enabled", False)):
        row.update(
            compute_expected_count_metrics(
                sample=sample,
                config=expected_count_cfg,
            )
        )

    mmd_cfg = metrics_cfg.get("mmd", {})
    if bool(mmd_cfg.get("enabled", False)):
        row.update(
            compute_mmd_metrics(
                sample=sample,
                config=mmd_cfg,
                prefix="mmd",
            )
        )
    
    depth_cfg = metrics_cfg.get("depth", {})
    if bool(depth_cfg.get("enabled", False)):
        row.update(
            compute_depth_metrics(
                sample=sample,
                config=depth_cfg,
            )
        )

    return row


def evaluate_models(
    h5_dirs: dict[str, str],
    *,
    metrics_cfg: dict[str, dict[str, Any]],
    max_samples_per_model: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all configured model runs.

    Returns:
        df_samples:
            one row per model and sample.
    """
    rows: list[dict[str, Any]] = []

    for model_name, h5_dir in h5_dirs.items():
        entries = list_h5_samples_for_model(model_name, h5_dir)

        if max_samples_per_model is not None:
            entries = entries[: int(max_samples_per_model)]

        if verbose:
            print(f"{model_name}: {len(entries)} samples from {h5_dir}")

        for idx, entry in enumerate(entries, start=1):
            if verbose and (idx == 1 or idx % 25 == 0 or idx == len(entries)):
                print(f"  {model_name}: {idx}/{len(entries)}")

            sample = load_h5_sample(entry)

            row: dict[str, Any] = {
                "model": model_name,
                "sample_id": entry.sample_id,
                "rank_file": entry.rank_file,
                "sample_key": entry.sample_key,
                "h5_path": entry.h5_path,
                "H": sample["shape"]["H"],
                "W": sample["shape"]["W"],
            }

            row.update(
                evaluate_sample(
                    sample,
                    metrics_cfg=metrics_cfg,
                )
            )

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _status_ok_fraction(series: pd.Series) -> float:
    """Fraction of samples whose metric status is 'ok'."""
    if len(series) == 0:
        return np.nan
    return float((series == "ok").mean())


def summarize_results(
    df_samples: pd.DataFrame,
    *,
    sort_by: str = "mmd_optimal_score_mean",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Aggregate sample-level metrics to one row per model.
    """
    if df_samples.empty:
        return pd.DataFrame()

    grouped = df_samples.groupby("model", dropna=False)

    agg_spec: dict[str, tuple[str, str]] = {
        "n_samples": ("sample_id", "count"),
    }

    if "expected_count" in df_samples.columns:
        agg_spec.update(
            {
                "expected_count_mean": ("expected_count", "mean"),
                "expected_count_std": ("expected_count", "std"),
                "gt_count_mean": ("gt_count", "mean"),
                "gt_count_std": ("gt_count", "std"),
                "count_error_mean": ("count_error", "mean"),
                "count_error_std": ("count_error", "std"),
                "count_abs_error_mean": ("count_abs_error", "mean"),
                "count_abs_error_median": ("count_abs_error", "median"),
                "count_abs_error_max": ("count_abs_error", "max"),
                "count_ratio_mean": ("count_ratio", "mean"),
                "count_ratio_std": ("count_ratio", "std"),
            }
        )

    if "mmd_optimal_score" in df_samples.columns:
        agg_spec.update(
            {
                "mmd_optimal_score_mean": ("mmd_optimal_score", "mean"),
                "mmd_optimal_score_std": ("mmd_optimal_score", "std"),
                "mmd_optimal_score_var": ("mmd_optimal_score", "var"),
                "mmd_optimal_score_median": ("mmd_optimal_score", "median"),
                "mmd_optimal_score_min": ("mmd_optimal_score", "min"),
                "mmd_optimal_score_max": ("mmd_optimal_score", "max"),
            }
        )
    
    if "depth_mae" in df_samples.columns:
        agg_spec.update(
            {
                "depth_mae_mean": ("depth_mae", "mean"),
                "depth_mae_std": ("depth_mae", "std"),
                "depth_rmse_mean": ("depth_rmse", "mean"),
                "depth_rmse_std": ("depth_rmse", "std"),
                "depth_bias_mean": ("depth_bias", "mean"),
                "depth_bias_std": ("depth_bias", "std"),
                "depth_abs_rel_error_mean": ("depth_abs_rel_error", "mean"),
                "depth_n_points_sum": ("depth_n_points", "sum"),
            }
        )
    
    if "depth_two_depth_used" in df_samples.columns:
        agg_spec.update(
            {
                "depth_two_depth_used_mean": ("depth_two_depth_used", "mean"),
            }
        )

    if "depth_far_choice_ratio" in df_samples.columns:
        agg_spec.update(
            {
                "depth_far_choice_ratio_mean": ("depth_far_choice_ratio", "mean"),
                "depth_far_choice_ratio_std": ("depth_far_choice_ratio", "std"),
            }
        )

    summary = grouped.agg(**agg_spec).reset_index()

    if "count_squared_error" in df_samples.columns:
        rmse = (
            grouped["count_squared_error"]
            .mean()
            .pow(0.5)
            .rename("count_rmse")
            .reset_index()
        )
        summary = summary.merge(rmse, on="model", how="left")

    if "mmd_status" in df_samples.columns:
        status = (
            grouped["mmd_status"]
            .apply(_status_ok_fraction)
            .rename("mmd_status_ok_fraction")
            .reset_index()
        )
        summary = summary.merge(status, on="model", how="left")

    if sort_by in summary.columns:
        summary = summary.sort_values(
            by=sort_by,
            ascending=ascending,
        ).reset_index(drop=True)

    return summary


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_comparison(
    h5_dirs: dict[str, str],
    *,
    metrics_cfg: dict[str, dict[str, Any]],
    output_dir: str | os.PathLike[str] = "./metric_comparison_results",
    max_samples_per_model: int | None = None,
    verbose: bool = True,
    sort_by: str = "mmd_optimal_score_mean",
    ascending: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the configured comparison and write CSV outputs.

    Returns:
        df_samples, df_summary
    """
    if not h5_dirs:
        raise ValueError("h5_dirs is empty. Add at least one model run.")

    if not metrics_cfg:
        raise ValueError("metrics_cfg is empty. Add at least one metric config.")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df_samples = evaluate_models(
        h5_dirs=h5_dirs,
        metrics_cfg=metrics_cfg,
        max_samples_per_model=max_samples_per_model,
        verbose=verbose,
    )

    df_summary = summarize_results(
        df_samples,
        sort_by=sort_by,
        ascending=ascending,
    )

    samples_path = output_dir / "sample_metrics.csv"
    summary_path = output_dir / "model_comparison_summary.csv"

    df_samples.to_csv(samples_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    if verbose:
        print("\nModel comparison summary:")
        print(df_summary.to_string(index=False))
        print(f"\nWrote sample metrics to: {samples_path}")
        print(f"Wrote summary metrics to: {summary_path}")

    return df_samples, df_summary


def run_from_config(
    config_path: str | os.PathLike[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load metric_eval_config.py and run the configured comparison.

    Notebook usage:
        from ppp_offline_metric_eval import run_from_config
        df_samples, df_summary = run_from_config("metric_eval_config.py")
    """
    cfg = load_python_config(config_path)

    h5_dirs = _get_cfg_attr(cfg, "H5_DIRS")
    metrics_cfg = _get_cfg_attr(cfg, "METRICS")
    output_dir = _get_cfg_attr(cfg, "OUTPUT_DIR", "./metric_comparison_results")
    max_samples = _get_cfg_attr(cfg, "MAX_SAMPLES_PER_MODEL", None)
    verbose = bool(_get_cfg_attr(cfg, "VERBOSE", True))
    sort_by = _get_cfg_attr(cfg, "SORT_BY", "mmd_optimal_score_mean")
    ascending = bool(_get_cfg_attr(cfg, "ASCENDING", False))

    if h5_dirs is None:
        raise ValueError("Config must define H5_DIRS.")

    if metrics_cfg is None:
        raise ValueError("Config must define METRICS.")

    return run_comparison(
        h5_dirs=h5_dirs,
        metrics_cfg=metrics_cfg,
        output_dir=output_dir,
        max_samples_per_model=max_samples,
        verbose=verbose,
        sort_by=sort_by,
        ascending=ascending,
    )


# ---------------------------------------------------------------------------
# Optional plotting helper
# ---------------------------------------------------------------------------

def plot_metric_boxplot(
    df_samples: pd.DataFrame,
    metric_col: str = "mmd_optimal_score",
    *,
    by: str = "model",
    rot: int = 45,
):
    """
    Quick notebook helper for boxplots.

    Example:
        plot_metric_boxplot(df_samples, "mmd_optimal_score")
    """
    if metric_col not in df_samples.columns:
        raise KeyError(f"Column not found in df_samples: {metric_col}")

    if by not in df_samples.columns:
        raise KeyError(f"Column not found in df_samples: {by}")

    ax = df_samples.boxplot(
        column=metric_col,
        by=by,
        rot=rot,
    )
    ax.set_title(metric_col)
    ax.figure.suptitle("")
    ax.set_xlabel(by)
    ax.set_ylabel(metric_col)
    return ax

def plot_expected_count_scatter(
    df_samples: pd.DataFrame,
    *,
    output_dir: str | os.PathLike[str] | None = None,
):
    """
    Create one scatterplot per model:

        x-axis: GT count
        y-axis: expected count
        dashed identity line: y = x

    If output_dir is given, plots are saved as PNG files.
    """
    import matplotlib.pyplot as plt

    required_cols = {"model", "gt_count", "expected_count"}
    missing = required_cols.difference(df_samples.columns)
    if missing:
        raise KeyError(f"df_samples is missing columns: {sorted(missing)}")

    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

    figures = {}

    for model_name, df_model in df_samples.groupby("model", dropna=False):
        fig, ax = plt.subplots(figsize=(6, 6))

        x = df_model["gt_count"].to_numpy()
        y = df_model["expected_count"].to_numpy()

        ax.scatter(x, y)

        if len(x) > 0 and len(y) > 0:
            xy_min = float(np.nanmin([x.min(), y.min()]))
            xy_max = float(np.nanmax([x.max(), y.max()]))

            ax.plot(
                [xy_min, xy_max],
                [xy_min, xy_max],
                linestyle="--",
            )

        ax.set_title(f"Expected Count vs GT Count\n{model_name}")
        ax.set_xlabel("GT count")
        ax.set_ylabel("Expected count")

        figures[model_name] = fig

        if output_path is not None:
            safe_name = str(model_name).replace("/", "_").replace("\\", "_")
            fig.savefig(output_path / f"{safe_name}_expected_count_scatter.png", bbox_inches="tight")

    return figures

if __name__ == "__main__":
    run_from_config("metric_eval_config.py")