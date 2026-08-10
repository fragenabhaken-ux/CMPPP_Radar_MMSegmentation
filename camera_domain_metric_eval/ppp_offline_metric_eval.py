"""Offline camera-domain marked-PPP model comparison."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from ppp_expected_count_tools import compute_expected_count_metrics
from ppp_mark_metric_tools import (
    compute_depth_metrics,
    compute_rcs_metrics,
    compute_velocity_metrics,
)
from ppp_metric_io_tools import list_h5_samples_for_model, load_h5_sample
from ppp_spatial_likelihood_tools import compute_likelihood_metrics


def load_python_config(path: str | os.PathLike[str]) -> ModuleType:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    spec = importlib.util.spec_from_file_location("camera_metric_eval_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enabled(metrics_cfg: dict[str, dict[str, Any]], key: str) -> bool:
    return bool(metrics_cfg.get(key, {}).get("enabled", False))


def evaluate_sample(
    sample: dict[str, Any],
    *,
    metrics_cfg: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if _enabled(metrics_cfg, "expected_count"):
        row.update(compute_expected_count_metrics(sample, metrics_cfg["expected_count"]))
    if _enabled(metrics_cfg, "likelihood"):
        row.update(compute_likelihood_metrics(sample, metrics_cfg["likelihood"]))
    if _enabled(metrics_cfg, "depth"):
        row.update(compute_depth_metrics(sample, metrics_cfg["depth"]))
    if _enabled(metrics_cfg, "velocity"):
        row.update(compute_velocity_metrics(sample, metrics_cfg["velocity"]))
    if _enabled(metrics_cfg, "rcs"):
        row.update(compute_rcs_metrics(sample, metrics_cfg["rcs"]))
    return row


def evaluate_models(
    h5_dirs: dict[str, str],
    *,
    metrics_cfg: dict[str, dict[str, Any]],
    max_samples_per_model: int | None = None,
    fail_fast: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_name, h5_dir in h5_dirs.items():
        entries = list_h5_samples_for_model(model_name, h5_dir)
        if max_samples_per_model is not None:
            entries = entries[: int(max_samples_per_model)]
        if verbose:
            print(f"{model_name}: {len(entries)} samples from {h5_dir}")

        for index, entry in enumerate(entries, start=1):
            if verbose and (index == 1 or index % 25 == 0 or index == len(entries)):
                print(f"  {model_name}: {index}/{len(entries)}")

            try:
                sample = load_h5_sample(entry)
                attrs = sample["attrs"]
                meta = sample["meta"]
                row: dict[str, Any] = {
                    "model": model_name,
                    "sample_uid": meta["sample_uid"],
                    "sample_uid_is_stable": meta["sample_uid_is_stable"],
                    "rank_file": meta["rank_file"],
                    "sample_key": meta["sample_key"],
                    "h5_path": meta["h5_path"],
                    "video_name": attrs.get("video_name", ""),
                    "video_frame_idx": attrs.get("video_frame_idx", np.nan),
                    "global_idx": attrs.get("global_idx", np.nan),
                    "H": sample["shape"]["H"],
                    "W": sample["shape"]["W"],
                }
                row.update(evaluate_sample(sample, metrics_cfg=metrics_cfg))
                row["evaluation_status"] = "ok"
                row["evaluation_error"] = ""
            except Exception as exc:  # noqa: BLE001 - preserved in result CSV
                if fail_fast:
                    raise RuntimeError(
                        f"Evaluation failed for {model_name}: "
                        f"{entry.h5_path}/{entry.sample_key}"
                    ) from exc
                row = {
                    "model": model_name,
                    "sample_uid": entry.fallback_sample_id,
                    "sample_uid_is_stable": False,
                    "rank_file": entry.rank_file,
                    "sample_key": entry.sample_key,
                    "h5_path": entry.h5_path,
                    "evaluation_status": "error",
                    "evaluation_error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)

    return pd.DataFrame(rows)


def validate_sample_alignment(
    df_samples: pd.DataFrame,
    *,
    require_stable_uid: bool = True,
) -> None:
    if df_samples.empty:
        raise ValueError("No samples were loaded")

    ok = df_samples[df_samples["evaluation_status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No samples were evaluated successfully")

    if require_stable_uid and not bool(ok["sample_uid_is_stable"].all()):
        bad = ok.loc[~ok["sample_uid_is_stable"], ["model", "rank_file", "sample_key"]]
        raise ValueError(
            "Stable sample identifiers are missing for some exports. "
            "The hook must export video_name/video_frame_idx or another stable ID.\n"
            + bad.head(10).to_string(index=False)
        )

    duplicates = ok.duplicated(["model", "sample_uid"], keep=False)
    if bool(duplicates.any()):
        raise ValueError(
            "Duplicate sample_uid values found within a model:\n"
            + ok.loc[duplicates, ["model", "sample_uid", "rank_file", "sample_key"]]
            .head(20)
            .to_string(index=False)
        )

    model_sets = {
        model: set(group["sample_uid"].astype(str))
        for model, group in ok.groupby("model", dropna=False)
    }
    reference_model = next(iter(model_sets))
    reference = model_sets[reference_model]
    mismatches: list[str] = []
    for model, values in model_sets.items():
        missing = sorted(reference - values)
        extra = sorted(values - reference)
        if missing or extra:
            mismatches.append(
                f"{model}: missing={len(missing)}, extra={len(extra)}, "
                f"missing_examples={missing[:3]}, extra_examples={extra[:3]}"
            )
    if mismatches:
        raise ValueError(
            f"Model exports do not contain the same samples as {reference_model}:\n"
            + "\n".join(mismatches)
        )


def _finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def _finite_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1)) if values.size > 1 else np.nan

def _finite_population_std(series: pd.Series) -> float:
    """Population SD over the evaluated samples."""
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    values = values[np.isfinite(values)]

    return (
        float(values.std(ddof=0))
        if values.size > 0
        else np.nan
    )

def _finite_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def _sum_numeric(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return 0.0
    return float(pd.to_numeric(group[column], errors="coerce").fillna(0.0).sum())


def _global_mark_metrics(group: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    n = int(round(_sum_numeric(group, f"{prefix}_n_points")))
    if n <= 0:
        return {
            f"{prefix}_mae_global": np.nan,
            f"{prefix}_rmse_global": np.nan,
            f"{prefix}_bias_global": np.nan,
            f"{prefix}_n_points_global": 0,
        }
    abs_sum = _sum_numeric(group, f"{prefix}_abs_error_sum")
    sq_sum = _sum_numeric(group, f"{prefix}_squared_error_sum")
    err_sum = _sum_numeric(group, f"{prefix}_error_sum")
    result: dict[str, float | int] = {
        f"{prefix}_mae_global": abs_sum / n,
        f"{prefix}_rmse_global": float(np.sqrt(sq_sum / n)),
        f"{prefix}_bias_global": err_sum / n,
        f"{prefix}_n_points_global": n,
    }
    abs_rel_col = f"{prefix}_abs_rel_error_sum"
    if abs_rel_col in group:
        result[f"{prefix}_abs_rel_error_global"] = _sum_numeric(group, abs_rel_col) / n
    return result


def summarize_results(
    df_samples: pd.DataFrame,
    *,
    sort_by: str = "likelihood_optimal_score_mean",
    ascending: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, all_group in df_samples.groupby("model", dropna=False):
        group = all_group[all_group["evaluation_status"] == "ok"].copy()
        row: dict[str, Any] = {
            "model": model,
            "n_samples": int(len(all_group)),
            "n_samples_ok": int(len(group)),
            "evaluation_ok_fraction": float(len(group) / len(all_group)) if len(all_group) else np.nan,
        }

        if "expected_count" in group:
            row.update(
                {
                    "expected_count_mean": _finite_mean(
                        group["expected_count"]
                    ),
                    "gt_count_mean": _finite_mean(
                        group["gt_count"]
                    ),

                    # Mean and SD of the signed count error.
                    "count_bias": _finite_mean(
                        group["count_error"]
                    ),
                    "count_error_std": _finite_population_std(
                        group["count_error"]
                    ),

                    "count_mae": _finite_mean(
                        group["count_abs_error"]
                    ),
                    "count_rmse": float(
                        np.sqrt(
                            _finite_mean(
                                group["count_squared_error"]
                            )
                        )
                    ),
                    "count_ratio_mean": _finite_mean(
                        group["count_ratio"]
                    ),
                }
            )

        if "likelihood_optimal_score" in group:
            row.update(
                {
                    "likelihood_optimal_score_mean": _finite_mean(group["likelihood_optimal_score"]),
                    "likelihood_optimal_score_std": _finite_std(group["likelihood_optimal_score"]),
                    "likelihood_optimal_score_median": _finite_median(group["likelihood_optimal_score"]),
                    "likelihood_optimal_sigma_px_mean": _finite_mean(group["likelihood_optimal_sigma_px"]),
                    "likelihood_optimal_sigma_px_std": _finite_std(group["likelihood_optimal_sigma_px"]),
                    "likelihood_score_sigma0_mean": _finite_mean(group["likelihood_score_sigma0"]),
                }
            )

        for prefix in ("depth", "velocity", "rcs"):
            mae_col = f"{prefix}_mae"
            if mae_col in group:
                row[f"{prefix}_mae_frame_mean"] = _finite_mean(group[mae_col])
                row[f"{prefix}_rmse_frame_mean"] = _finite_mean(group[f"{prefix}_rmse"])
                row[f"{prefix}_bias_frame_mean"] = _finite_mean(group[f"{prefix}_bias"])
                row.update(_global_mark_metrics(group, prefix))

        rows.append(row)

    summary = pd.DataFrame(rows)
    if sort_by in summary.columns:
        summary = summary.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    return summary

def bootstrap_summary_confidence_intervals(
    df_samples: pd.DataFrame,
    metrics: list[str],
    num_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 20260731,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute independent bootstrap confidence intervals for each model.

    Each model is bootstrapped from its own successfully evaluated samples.
    Sample IDs and sample counts do not need to match between models.
    """
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie strictly between 0 and 1."
        )

    if not metrics:
        raise ValueError("metrics must not be empty.")

    if df_samples.empty:
        return pd.DataFrame(), pd.DataFrame()

    successful = df_samples.copy()

    if "evaluation_status" in successful.columns:
        successful = successful[
            successful["evaluation_status"] == "ok"
        ].copy()

    if successful.empty:
        raise ValueError(
            "No successfully evaluated samples are available "
            "for bootstrapping."
        )

    models = list(successful["model"].dropna().unique())

    seed_sequences = np.random.SeedSequence(seed).spawn(len(models))

    replicate_rows: list[dict[str, Any]] = []

    for model, model_seed in zip(models, seed_sequences):
        model_table = (
            successful[successful["model"] == model]
            .reset_index(drop=True)
        )

        n_samples = len(model_table)

        if n_samples == 0:
            continue

        rng = np.random.default_rng(model_seed)

        for bootstrap_index in range(num_resamples):
            sampled_positions = rng.integers(
                low=0,
                high=n_samples,
                size=n_samples,
            )

            sampled_table = (
                model_table.iloc[sampled_positions]
                .reset_index(drop=True)
            )

            sampled_summary = summarize_results(
                sampled_table,
                sort_by=None,
                ascending=False,
            )

            if sampled_summary.empty:
                continue

            summary_row = sampled_summary.iloc[0]

            replicate_row: dict[str, Any] = {
                "model": model,
                "bootstrap_index": bootstrap_index,
                "bootstrap_n_samples": n_samples,
            }

            for metric in metrics:
                if metric not in summary_row.index:
                    raise ValueError(
                        f"Bootstrap metric {metric!r} is not produced by "
                        f"summarize_results(). Available columns: "
                        f"{sorted(sampled_summary.columns)}"
                    )

                replicate_row[metric] = pd.to_numeric(
                    summary_row[metric],
                    errors="coerce",
                )

            replicate_rows.append(replicate_row)

    df_replicates = pd.DataFrame(replicate_rows)

    alpha = 1.0 - confidence_level
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - alpha / 2.0

    ci_rows: list[dict[str, Any]] = []

    for model in models:
        model_replicates = df_replicates[
            df_replicates["model"] == model
        ]

        ci_row: dict[str, Any] = {
            "model": model,
            "bootstrap_num_resamples": len(model_replicates),
            "bootstrap_confidence_level": confidence_level,
        }

        for metric in metrics:
            values = pd.to_numeric(
                model_replicates[metric],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            values = values[np.isfinite(values)]

            if len(values) == 0:
                lower = np.nan
                upper = np.nan
            else:
                lower, upper = np.quantile(
                    values,
                    [lower_quantile, upper_quantile],
                )

            ci_row[f"{metric}_ci95_low"] = float(lower)
            ci_row[f"{metric}_ci95_high"] = float(upper)

        ci_rows.append(ci_row)

    return pd.DataFrame(ci_rows), df_replicates

def run_comparison(
    h5_dirs: dict[str, str],
    *,
    metrics_cfg: dict[str, dict[str, Any]],
    output_dir: str | os.PathLike[str],
    max_samples_per_model: int | None = None,
    fail_fast: bool = True,
    strict_sample_match: bool = True,
    require_stable_uid: bool = True,
    verbose: bool = True,
    sort_by: str = "likelihood_optimal_score_mean",
    ascending: bool = False,
    bootstrap_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    df_samples = evaluate_models(
        h5_dirs,
        metrics_cfg=metrics_cfg,
        max_samples_per_model=max_samples_per_model,
        fail_fast=fail_fast,
        verbose=verbose,
    )
    if strict_sample_match:
        validate_sample_alignment(df_samples, require_stable_uid=require_stable_uid)

    df_samples = df_samples.sort_values(["model", "sample_uid"]).reset_index(drop=True)
    df_summary = summarize_results(df_samples, sort_by=sort_by, ascending=ascending)
    df_bootstrap_ci = pd.DataFrame()
    df_bootstrap_replicates = pd.DataFrame()

    if bootstrap_config and bool(
        bootstrap_config.get(
            "enabled",
            True,
        )
    ):
        bootstrap_metrics = bootstrap_config.get(
            "metrics"
        )

        if not bootstrap_metrics:
            raise ValueError(
                "BOOTSTRAP must define a non-empty 'metrics' list."
            )

        (
            df_bootstrap_ci,
            df_bootstrap_replicates,
        ) = bootstrap_summary_confidence_intervals(
            df_samples,
            metrics=list(bootstrap_metrics),
            num_resamples=int(
                bootstrap_config.get(
                    "num_resamples",
                    5000,
                )
            ),
            confidence_level=float(
                bootstrap_config.get(
                    "confidence_level",
                    0.95,
                )
            ),
            seed=int(
                bootstrap_config.get(
                    "seed",
                    20260731,
                )
            ),
        )

        df_summary = df_summary.merge(
            df_bootstrap_ci,
            on="model",
            how="left",
            validate="one_to_one",
        )
    
    samples_path = output_path / "sample_metrics.csv"
    summary_path = output_path / "model_comparison_summary.csv"
    manifest_path = output_path / "sample_manifest.csv"
    df_samples.to_csv(
        samples_path,
        index=False,
    )
    df_summary.to_csv(
        summary_path,
        index=False,
    )
    df_samples[
        [
            "model",
            "sample_uid",
            "video_name",
            "video_frame_idx",
        ]
    ].to_csv(
        manifest_path,
        index=False,
    )

    if not df_bootstrap_ci.empty:
        df_bootstrap_ci.to_csv(
            output_path / "bootstrap_confidence_intervals.csv",
            index=False,
        )

    if not df_bootstrap_replicates.empty:
        df_bootstrap_replicates.to_csv(
            output_path / "bootstrap_replicates.csv",
            index=False,
        )

    if verbose:
        print("\nModel comparison summary:")
        print(df_summary.to_string(index=False))
        print(f"\nWrote: {samples_path}")
        print(f"Wrote: {summary_path}")
        print(f"Wrote: {manifest_path}")
    return df_samples, df_summary


def run_from_config(path: str | os.PathLike[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_python_config(path)
    return run_comparison(
        h5_dirs=cfg.H5_DIRS,
        metrics_cfg=cfg.METRICS,
        output_dir=getattr(cfg, "OUTPUT_DIR", "./camera_metric_results"),
        max_samples_per_model=getattr(cfg, "MAX_SAMPLES_PER_MODEL", None),
        fail_fast=bool(getattr(cfg, "FAIL_FAST", True)),
        strict_sample_match=bool(getattr(cfg, "STRICT_SAMPLE_MATCH", True)),
        require_stable_uid=bool(getattr(cfg, "REQUIRE_STABLE_UID", True)),
        verbose=bool(getattr(cfg, "VERBOSE", True)),
        sort_by=getattr(cfg, "SORT_BY", "likelihood_optimal_score_mean",),
        ascending=bool(getattr(cfg, "ASCENDING", False,)),
        bootstrap_config=getattr(cfg, "BOOTSTRAP", None,),
        )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="metric_eval_config.py")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
