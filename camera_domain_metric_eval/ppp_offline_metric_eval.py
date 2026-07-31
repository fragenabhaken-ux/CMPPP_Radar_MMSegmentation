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
    *,
    metrics: list[str] | tuple[str, ...],
    num_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 20260731,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired percentile-bootstrap confidence intervals.

    Evaluation samples are drawn with replacement. The same sampled
    positions are used for every model in each bootstrap repetition,
    preserving the paired comparison between models.

    The complete model summary is recomputed for every bootstrap sample.
    This correctly handles nonlinear or weighted statistics such as count
    RMSE and global point-weighted mark MAEs.
    """
    if num_resamples <= 0:
        raise ValueError(
            "num_resamples must be positive."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie strictly between 0 and 1."
        )

    successful = df_samples[
        df_samples["evaluation_status"] == "ok"
    ].copy()

    if successful.empty:
        raise ValueError(
            "Cannot bootstrap because no samples were evaluated "
            "successfully."
        )

    # Paired bootstrapping requires stable physical-frame identifiers and
    # exactly the same evaluation samples for every model.
    validate_sample_alignment(
        successful,
        require_stable_uid=True,
    )

    models = successful[
        "model"
    ].drop_duplicates().tolist()

    if not models:
        raise ValueError(
            "No models found in df_samples."
        )

    model_tables: dict[str, pd.DataFrame] = {}

    for model in models:
        table = successful[
            successful["model"] == model
        ].copy()

        table["sample_uid"] = table[
            "sample_uid"
        ].astype(str)

        if bool(table["sample_uid"].duplicated().any()):
            duplicates = table.loc[
                table["sample_uid"].duplicated(keep=False),
                "sample_uid",
            ].tolist()

            raise ValueError(
                f"Model {model!r} contains duplicate sample UIDs: "
                f"{duplicates[:5]}"
            )

        model_tables[model] = table.set_index(
            "sample_uid",
            drop=False,
        )

    reference_model = models[0]
    reference_uids = sorted(
        model_tables[reference_model].index.tolist()
    )
    reference_uid_set = set(reference_uids)

    for model in models[1:]:
        model_uid_set = set(
            model_tables[model].index.tolist()
        )

        if model_uid_set != reference_uid_set:
            missing = sorted(
                reference_uid_set - model_uid_set
            )
            extra = sorted(
                model_uid_set - reference_uid_set
            )

            raise ValueError(
                "Paired bootstrapping requires identical samples "
                "for every model. "
                f"Model {model!r}: missing={len(missing)}, "
                f"extra={len(extra)}, "
                f"first_missing={missing[:5]}, "
                f"first_extra={extra[:5]}"
            )

    # Put every model in exactly the same physical-sample order.
    aligned_tables = {
        model: (
            model_tables[model]
            .loc[reference_uids]
            .reset_index(drop=True)
        )
        for model in models
    }

    original_summary = summarize_results(
        successful,
        sort_by="",
        ascending=False,
    )

    missing_metrics = [
        metric
        for metric in metrics
        if metric not in original_summary.columns
    ]

    if missing_metrics:
        raise KeyError(
            "The following requested bootstrap metrics are not "
            "produced by summarize_results(): "
            f"{missing_metrics}"
        )

    rng = np.random.default_rng(seed)
    num_samples = len(reference_uids)

    replicate_rows: list[dict[str, Any]] = []

    for bootstrap_index in range(num_resamples):
        # Draw N sample positions from N original samples, with replacement.
        sampled_positions = rng.integers(
            low=0,
            high=num_samples,
            size=num_samples,
        )

        # Use the same bootstrap draw for every model.
        bootstrap_samples = pd.concat(
            [
                aligned_tables[model]
                .iloc[sampled_positions]
                .copy()
                for model in models
            ],
            ignore_index=True,
        )

        bootstrap_summary = summarize_results(
            bootstrap_samples,
            sort_by="",
            ascending=False,
        )

        for _, summary_row in bootstrap_summary.iterrows():
            replicate: dict[str, Any] = {
                "bootstrap_index": bootstrap_index,
                "model": summary_row["model"],
            }

            for metric in metrics:
                value = summary_row.get(
                    metric,
                    np.nan,
                )

                replicate[metric] = (
                    float(value)
                    if pd.notna(value)
                    else np.nan
                )

            replicate_rows.append(replicate)

    df_replicates = pd.DataFrame(
        replicate_rows
    )

    alpha = (
        1.0 - confidence_level
    ) / 2.0

    confidence_percent = int(
        round(confidence_level * 100.0)
    )

    ci_rows: list[dict[str, Any]] = []

    for model, group in df_replicates.groupby(
        "model",
        sort=False,
    ):
        ci_row: dict[str, Any] = {
            "model": model,
        }

        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            finite = values[
                np.isfinite(values)
            ]

            low_name = (
                f"{metric}_ci{confidence_percent}_low"
            )
            high_name = (
                f"{metric}_ci{confidence_percent}_high"
            )
            se_name = (
                f"{metric}_bootstrap_se"
            )
            n_name = (
                f"{metric}_bootstrap_n_valid"
            )

            if finite.size == 0:
                ci_row[low_name] = np.nan
                ci_row[high_name] = np.nan
                ci_row[se_name] = np.nan
                ci_row[n_name] = 0
                continue

            ci_row[low_name] = float(
                np.quantile(
                    finite,
                    alpha,
                )
            )
            ci_row[high_name] = float(
                np.quantile(
                    finite,
                    1.0 - alpha,
                )
            )
            ci_row[se_name] = (
                float(finite.std(ddof=1))
                if finite.size > 1
                else np.nan
            )
            ci_row[n_name] = int(
                finite.size
            )

        ci_rows.append(ci_row)

    df_ci = pd.DataFrame(ci_rows)

    return df_ci, df_replicates

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
