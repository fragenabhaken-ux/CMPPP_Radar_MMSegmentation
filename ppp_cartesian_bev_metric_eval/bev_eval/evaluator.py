from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from .count_metrics import compute_count_metrics
from .io import H5Entry, geometry_signature, index_model_samples, load_bev_sample
from .likelihood import compute_likelihood_metrics
from .mark_metrics import compute_mark_metrics
from .pointcloud import ground_truth_point_cloud, sample_predicted_ppp, stable_seed
from .radargen_metrics import compute_radargen_pointcloud_metrics
from .types import BEVGeometry


def load_python_config(config_path: str | os.PathLike[str]) -> ModuleType:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    spec = importlib.util.spec_from_file_location("bev_metric_eval_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_attr(module: ModuleType, name: str, default: Any = None) -> Any:
    return getattr(module, name, default)


def _validate_expected_geometry(
    geometry: BEVGeometry,
    expected: dict[str, Any] | None,
) -> None:
    if not expected:
        return
    aliases = {
        "x_min": "bev_x_min",
        "x_max": "bev_x_max",
        "y_min": "bev_y_min",
        "y_max": "bev_y_max",
        "height": "bev_height",
        "width": "bev_width",
    }
    actual = geometry.as_dict()
    for short, full in aliases.items():
        if full not in expected and short not in expected:
            continue
        wanted = expected.get(full, expected.get(short))
        got = actual[full]
        if isinstance(got, int):
            matches = int(wanted) == got
        else:
            matches = np.isclose(float(wanted), float(got), rtol=0.0, atol=1e-9)
        if not matches:
            raise ValueError(
                f"Unexpected BEV geometry for {full}: expected {wanted}, got {got}."
            )


def _nanmean_std(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan, np.nan, 0
    return float(finite.mean()), float(finite.std(ddof=0)), int(finite.size)


def _sampling_methods(config: dict[str, Any]) -> tuple[str, ...]:
    configured = config.get(
        "sampling_methods",
        config.get("sampling_method", ("thinning", "block")),
    )
    if isinstance(configured, str):
        if configured.lower() == "both":
            methods = ("thinning", "block")
        else:
            methods = (configured.lower(),)
    else:
        methods = tuple(str(value).lower() for value in configured)

    if not methods:
        raise ValueError("At least one RadarGen sampling method is required.")
    if len(set(methods)) != len(methods):
        raise ValueError(f"Duplicate sampling methods are not allowed: {methods}.")
    unsupported = [method for method in methods if method not in {"thinning", "block"}]
    if unsupported:
        raise ValueError(
            f"Unsupported sampling methods {unsupported}. Use 'thinning' and/or 'block'."
        )
    return methods


def compute_radargen_mc_metrics(
    sample: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    num_draws = int(config.get("num_draws", 1))
    if num_draws <= 0:
        raise ValueError("RadarGen num_draws must be > 0.")

    methods = _sampling_methods(config)
    base_seed = int(config.get("seed", 20260726))
    block_size = int(config.get("block_size", 8))
    block_placement = str(config.get("block_placement", "weighted")).lower()
    if block_size <= 0:
        raise ValueError("block_size must be > 0.")
    if block_placement not in {"weighted", "uniform"}:
        raise ValueError("block_placement must be 'weighted' or 'uniform'.")

    max_sampled_points = config.get("max_sampled_points")
    if max_sampled_points is not None:
        max_sampled_points = int(max_sampled_points)
        if max_sampled_points <= 0:
            raise ValueError("max_sampled_points must be positive or None.")

    gt = ground_truth_point_cloud(sample)
    sample_uid = str(sample["meta"]["sample_uid"])
    doppler_period = sample["attrs"].get("doppler_period")
    if doppler_period is not None:
        doppler_period = float(doppler_period)

    result: dict[str, Any] = {
        "radargen_num_draws": num_draws,
        "radargen_sampling_seed": base_seed,
        "radargen_sampling_methods": ",".join(methods),
        "radargen_block_size": block_size,
        "radargen_block_placement": block_placement,
        "pointcloud_gt_count": gt.n_points,
    }

    for method in methods:
        draw_rows: list[dict[str, float | int]] = []
        for draw_index in range(num_draws):
            # Include the method in the stream while retaining common random
            # numbers across model runs for the same sample and method.
            method_uid = f"{sample_uid}|sampling_method={method}"
            rng = np.random.default_rng(
                stable_seed(base_seed, method_uid, draw_index)
            )
            pred = sample_predicted_ppp(
                sample,
                rng=rng,
                method=method,
                block_size=block_size,
                block_placement=block_placement,
                max_points=max_sampled_points,
            )
            draw_rows.append(
                compute_radargen_pointcloud_metrics(
                    pred,
                    gt,
                    sample["geometry"],
                    doppler_period=doppler_period,
                    config=config,
                )
            )

        metric_prefix = f"radargen_{method}"
        keys = sorted({key for row in draw_rows for key in row})
        for key in keys:
            values = [float(row[key]) for row in draw_rows if key in row]
            mean, std, n_valid = _nanmean_std(values)
            result[f"{metric_prefix}_{key}"] = mean
            result[f"{metric_prefix}_{key}_mc_std"] = std
            result[f"{metric_prefix}_{key}_mc_valid_draws"] = n_valid

        result[f"{metric_prefix}_nonempty_pair_draw_fraction"] = float(
            np.mean([row["pointcloud_pair_nonempty"] for row in draw_rows])
        )

    return result


def evaluate_sample(
    sample: dict[str, Any],
    *,
    metrics_config: dict[str, dict[str, Any]],
    expected_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_expected_geometry(sample["geometry"], expected_geometry)
    row: dict[str, Any] = {}

    count_cfg = metrics_config.get("count", {})
    if bool(count_cfg.get("enabled", True)):
        row.update(compute_count_metrics(sample, count_cfg))

    likelihood_cfg = metrics_config.get("likelihood", {})
    if bool(likelihood_cfg.get("enabled", True)):
        row.update(compute_likelihood_metrics(sample, likelihood_cfg))

    marks_cfg = metrics_config.get("marks", {})
    if bool(marks_cfg.get("enabled", True)):
        row.update(compute_mark_metrics(sample, marks_cfg))

    radargen_cfg = metrics_config.get("radargen", {})
    if bool(radargen_cfg.get("enabled", True)):
        row.update(compute_radargen_mc_metrics(sample, radargen_cfg))

    return row


def _aligned_uids(
    indexes: dict[str, dict[str, H5Entry]],
    *,
    mode: str,
) -> list[str]:
    mode = str(mode).lower()
    sets = {model: set(index) for model, index in indexes.items()}
    if not sets:
        return []

    models = list(sets)
    reference = sets[models[0]]
    if mode == "strict":
        problems = []
        for model in models[1:]:
            missing = sorted(reference - sets[model])
            extra = sorted(sets[model] - reference)
            if missing or extra:
                problems.append(
                    f"{model}: missing={len(missing)}, extra={len(extra)}, "
                    f"first_missing={missing[:3]}, first_extra={extra[:3]}"
                )
        if problems:
            raise ValueError(
                "Model exports do not contain identical stable sample IDs. "
                "Use ALIGNMENT_MODE='intersection' only when intentional.\n"
                + "\n".join(problems)
            )
        return sorted(reference)

    if mode == "intersection":
        common = set.intersection(*sets.values())
        if not common:
            raise ValueError("Model exports have no common stable sample IDs.")
        return sorted(common)

    if mode == "none":
        return []

    raise ValueError("ALIGNMENT_MODE must be 'strict', 'intersection', or 'none'.")


def _base_row(sample: dict[str, Any]) -> dict[str, Any]:
    geometry: BEVGeometry = sample["geometry"]
    attrs = sample["attrs"]
    row: dict[str, Any] = {
        "model": sample["meta"]["model"],
        "sample_uid": sample["meta"]["sample_uid"],
        "rank_file": sample["meta"]["rank_file"],
        "sample_key": sample["meta"]["sample_key"],
        "h5_path": sample["meta"]["h5_path"],
        "source_prediction_domain": attrs.get("source_prediction_domain"),
        "source_ground_truth_domain": attrs.get("source_ground_truth_domain"),
        "video_name": attrs.get("video_name"),
        "video_frame_idx": attrs.get("video_frame_idx"),
        "global_idx": attrs.get("global_idx", attrs.get("frame_idx")),
        "doppler_period": attrs.get("doppler_period"),
    }
    row.update(geometry.as_dict())
    return row


def evaluate_models(
    h5_dirs: dict[str, str],
    *,
    metrics_config: dict[str, dict[str, Any]],
    expected_geometry: dict[str, Any] | None = None,
    alignment_mode: str = "strict",
    max_samples_per_model: int | None = None,
    fail_fast: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not h5_dirs:
        raise ValueError("H5_DIRS is empty.")

    indexes = {
        model: index_model_samples(model, directory)
        for model, directory in h5_dirs.items()
    }
    aligned = _aligned_uids(indexes, mode=alignment_mode)
    if max_samples_per_model is not None:
        max_n = int(max_samples_per_model)
        if max_n <= 0:
            raise ValueError("MAX_SAMPLES_PER_MODEL must be positive or None.")
        if aligned:
            aligned = aligned[:max_n]

    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    reference_geometry: tuple[float, float, float, float, int, int] | None = None

    for model, index in indexes.items():
        if aligned:
            uids = aligned
        else:
            uids = sorted(index)
            if max_samples_per_model is not None:
                uids = uids[: int(max_samples_per_model)]

        if verbose:
            print(f"{model}: evaluating {len(uids)} samples")

        for position, uid in enumerate(uids, start=1):
            entry = index[uid]
            if verbose and (position == 1 or position % 25 == 0 or position == len(uids)):
                print(f"  {model}: {position}/{len(uids)}")
            try:
                sample = load_bev_sample(entry)
                signature = geometry_signature(sample["geometry"])
                if reference_geometry is None:
                    reference_geometry = signature
                elif signature != reference_geometry:
                    raise ValueError(
                        "All compared exports must use identical Cartesian BEV geometry. "
                        f"Reference={reference_geometry}, current={signature}."
                    )

                row = _base_row(sample)
                row.update(
                    evaluate_sample(
                        sample,
                        metrics_config=metrics_config,
                        expected_geometry=expected_geometry,
                    )
                )
                row["evaluation_status"] = "ok"
                row["evaluation_error"] = ""
            except Exception as exc:
                if fail_fast:
                    raise
                row = {
                    "model": model,
                    "sample_uid": uid,
                    "rank_file": entry.rank_file,
                    "sample_key": entry.sample_key,
                    "h5_path": entry.h5_path,
                    "evaluation_status": "error",
                    "evaluation_error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            manifest_rows.append(
                {
                    "model": model,
                    "sample_uid": uid,
                    "rank_file": entry.rank_file,
                    "sample_key": entry.sample_key,
                    "h5_path": entry.h5_path,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(manifest_rows)


def _status_ok_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    return float((series == "ok").mean())


def summarize_results(
    df_samples: pd.DataFrame,
    *,
    sort_by: str | None = "likelihood_optimal_score_mean",
    ascending: bool = False,
) -> pd.DataFrame:
    if df_samples.empty:
        return pd.DataFrame()

    successful = df_samples[df_samples["evaluation_status"] == "ok"].copy()
    if successful.empty:
        return pd.DataFrame(
            {
                "model": sorted(df_samples["model"].unique()),
                "n_samples_ok": 0,
            }
        )

    excluded_numeric = {
        "video_frame_idx",
        "global_idx",
        "bev_height",
        "bev_width",
        "radargen_sampling_seed",
        "velocity_gt_abs_min",
    }
    sufficient_stat_suffixes = (
        "_abs_error_sum",
        "_squared_error_sum",
        "_signed_error_sum",
        "_mc_valid_draws",
    )

    summaries: list[dict[str, Any]] = []
    for model, group in successful.groupby("model", sort=False):
        out: dict[str, Any] = {
            "model": model,
            "n_samples_ok": int(len(group)),
            "n_samples_total": int((df_samples["model"] == model).sum()),
        }

        for column in group.columns:
            if column in excluded_numeric or column in {
                "model",
                "sample_uid",
                "rank_file",
                "sample_key",
                "h5_path",
                "evaluation_status",
                "evaluation_error",
            }:
                continue
            if column.startswith("bev_"):
                continue
            if column.endswith(sufficient_stat_suffixes):
                continue
            if not pd.api.types.is_numeric_dtype(group[column]):
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                out[f"{column}_mean"] = np.nan
                out[f"{column}_std"] = np.nan
                out[f"{column}_median"] = np.nan
                out[f"{column}_n_valid"] = 0
            else:
                out[f"{column}_mean"] = float(finite.mean())
                out[f"{column}_std"] = float(finite.std(ddof=0))
                out[f"{column}_median"] = float(finite.median())
                out[f"{column}_n_valid"] = int(len(finite))

        if "count_squared_error" in group:
            out["count_rmse"] = float(np.sqrt(group["count_squared_error"].mean()))

        for prefix in ("velocity", "velocity_nonzero", "rcs"):
            n_col = f"{prefix}_n_points"
            abs_col = f"{prefix}_abs_error_sum"
            sq_col = f"{prefix}_squared_error_sum"
            signed_col = f"{prefix}_signed_error_sum"

            required_columns = (
                n_col,
                abs_col,
                sq_col,
                signed_col,
            )

            if all(col in group.columns for col in required_columns):
                n_points = float(group[n_col].sum())
                out[f"{prefix}_n_points_total"] = int(n_points)

                if n_points > 0.0:
                    out[f"{prefix}_mae_global"] = float(
                        group[abs_col].sum() / n_points
                    )
                    out[f"{prefix}_rmse_global"] = float(
                        np.sqrt(group[sq_col].sum() / n_points)
                    )
                    out[f"{prefix}_bias_global"] = float(
                        group[signed_col].sum() / n_points
                    )
                else:
                    out[f"{prefix}_mae_global"] = np.nan
                    out[f"{prefix}_rmse_global"] = np.nan
                    out[f"{prefix}_bias_global"] = np.nan

        summaries.append(out)

    summary = pd.DataFrame(summaries)
    if sort_by and sort_by in summary.columns:
        summary = summary.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    return summary

def bootstrap_summary_confidence_intervals(
    df_samples: pd.DataFrame,
    *,
    metrics: list[str],
    num_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 20260731,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute independent bootstrap confidence intervals for each model.

    Models do not need to contain the same sample IDs or the same number
    of successful samples.
    """
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between 0 and 1.")

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
            "No successfully evaluated samples are available for bootstrap."
        )

    models = list(successful["model"].dropna().unique())

    # Deterministic but independent random stream for every model.
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
                model_table
                .iloc[sampled_positions]
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
                        f"summarize_results(). Available columns include: "
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
            "bootstrap_num_resamples": int(len(model_replicates)),
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

    df_ci = pd.DataFrame(ci_rows)

    return df_ci, df_replicates

def run_evaluation(
    h5_dirs: dict[str, str],
    *,
    metrics_config: dict[str, dict[str, Any]],
    output_dir: str | os.PathLike[str],
    expected_geometry: dict[str, Any] | None = None,
    alignment_mode: str = "strict",
    max_samples_per_model: int | None = None,
    bootstrap_config: dict[str, Any] | None = None,
    fail_fast: bool = True,
    verbose: bool = True,
    sort_by: str | None = "likelihood_optimal_score_mean",
    ascending: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    df_samples, df_manifest = evaluate_models(
        h5_dirs,
        metrics_config=metrics_config,
        expected_geometry=expected_geometry,
        alignment_mode=alignment_mode,
        max_samples_per_model=max_samples_per_model,
        fail_fast=fail_fast,
        verbose=verbose,
    )
    df_summary = summarize_results(
        df_samples,
        sort_by=sort_by,
        ascending=ascending,
    )

    df_bootstrap_replicates = pd.DataFrame()

    if bootstrap_config and bool(
        bootstrap_config.get("enabled", True)
    ):
        bootstrap_metrics = bootstrap_config.get("metrics")

        if not bootstrap_metrics:
            raise ValueError(
                "Bootstrap configuration must define a non-empty "
                "'metrics' list."
            )

        df_bootstrap_ci, df_bootstrap_replicates = (
            bootstrap_summary_confidence_intervals(
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
        )

        df_summary = df_summary.merge(
            df_bootstrap_ci,
            on="model",
            how="left",
            validate="one_to_one",
        )

    df_samples.to_csv(output_path / "sample_metrics.csv", index=False)
    df_summary.to_csv(output_path / "model_summary.csv", index=False)
    df_manifest.to_csv(output_path / "sample_manifest.csv", index=False)
    if not df_bootstrap_replicates.empty:
        df_bootstrap_replicates.to_csv(
            output_path / "bootstrap_replicates.csv",
            index=False,
        )

    if verbose:
        print("\nModel summary:")
        print(df_summary.to_string(index=False))
        print(f"\nOutputs written to {output_path}")

    return df_samples, df_summary


def run_from_config(
    config_path: str | os.PathLike[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_python_config(config_path)
    h5_dirs = _config_attr(cfg, "H5_DIRS")
    metrics_config = _config_attr(cfg, "METRICS")
    if h5_dirs is None:
        raise ValueError("Config must define H5_DIRS.")
    if metrics_config is None:
        raise ValueError("Config must define METRICS.")

    output_dir = _config_attr(cfg, "OUTPUT_DIR", "./bev_metric_results")
    df_samples, df_summary = run_evaluation(
        h5_dirs=h5_dirs,
        metrics_config=metrics_config,
        output_dir=output_dir,
        expected_geometry=_config_attr(cfg, "EXPECTED_BEV_GEOMETRY", None),
        alignment_mode=_config_attr(cfg, "ALIGNMENT_MODE", "strict"),
        max_samples_per_model=_config_attr(cfg, "MAX_SAMPLES_PER_MODEL", None),
        fail_fast=bool(_config_attr(cfg, "FAIL_FAST", True)),
        verbose=bool(_config_attr(cfg, "VERBOSE", True)),
        sort_by=_config_attr(cfg, "SORT_BY", "likelihood_optimal_score_mean"),
        ascending=bool(_config_attr(cfg, "ASCENDING", False)),
        bootstrap_config=_config_attr(cfg, "BOOTSTRAP", None),
    )

    # Save a readable snapshot of user-facing config values.
    snapshot = {
        name: repr(getattr(cfg, name))
        for name in dir(cfg)
        if name.isupper() and not name.startswith("__")
    }
    output_path = Path(output_dir).expanduser().resolve()
    with open(output_path / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)

    return df_samples, df_summary
