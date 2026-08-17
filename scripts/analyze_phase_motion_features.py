#!/usr/bin/env python3
"""Analyse phase-specific motion without treating cycles as independent samples.

The dense phase labels are aligned to the original bilateral position traces.  Features
are first calculated within a cycle and phase, then averaged to one row per recording.
Inference and prediction therefore use recordings, not frames or cycles, as the unit of
analysis.  Transfer features are exported but excluded from cross-cohort modelling because
the Urology 1 annotation protocol folded transfer into placement for some recordings.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

import matplotlib

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "paper" / "build" / "mplconfig"))
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Features .* are constant")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUT = PROJECT / "data" / "derived"
FIG = PROJECT / "paper" / "figures"
TAB = PROJECT / "paper" / "tables"
for directory in (OUT, FIG, TAB):
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260618
COHORT_ORDER = ["BAPES2024", "6DOF2023", "7DOF2024"]
COHORT_LABELS = {
    "BAPES2024": "Paediatric",
    "6DOF2023": "Urology 1",
    "7DOF2024": "Urology 2",
}
COHORT_COLOURS = {
    "BAPES2024": "#009E73",
    "6DOF2023": "#D55E00",
    "7DOF2024": "#4C78A8",
}
BASE_PHASES = ["reach", "grasp", "transfer", "place"]
PHASES = [*BASE_PHASES, "transport_place"]


def first_existing(candidates: list[Path], kind: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate {kind}. Checked:\n  {joined}")


def locate_inputs() -> tuple[Path, Path]:
    dev_root = Path.home() / "dev" / "PhD" / "AI-ELT"
    phase_cache = first_existing(
        [
            Path(os.environ["LASK_PHASE_CACHE"]).expanduser()
            if "LASK_PHASE_CACHE" in os.environ
            else Path("/__unset_phase_cache__"),
            dev_root / "phase_recognition" / "outputs" / "cache",
            PROJECT.parent / "BTPN-MT" / "data" / "cache",
        ],
        "phase cache",
    )
    origin_motion = first_existing(
        [
            Path(os.environ["LASK_ORIGIN_MOTION"]).expanduser()
            if "LASK_ORIGIN_MOTION" in os.environ
            else Path("/__unset_origin_motion__"),
            dev_root / "outputs" / "ssl" / "origin" / "origin_data" / "ORIGIN_ALL",
            PROJECT.parent
            / "AI-ELT"
            / "outputs"
            / "ssl"
            / "origin"
            / "origin_data"
            / "ORIGIN_ALL",
        ],
        "aligned kinematic cache",
    )
    return phase_cache, origin_motion


def phase_trial_key(dataset: str, trial_short: str) -> str:
    match = re.search(r"(?:Trial|Test)\s*(\d+)", str(trial_short))
    if not match:
        return f"{dataset}_{trial_short}"
    number = int(match.group(1))
    if dataset == "6DOF2023":
        return f"6Test {number}"
    if dataset == "7DOF2024":
        return f"7Trial{number}"
    if dataset == "BAPES2024":
        return f"BTrial{number}"
    return f"{dataset}_{number}"


def fps_for(dataset: str) -> int:
    return 26 if dataset == "6DOF2023" else 13


def smooth_positions(position: np.ndarray, fps: int, window_s: float = 0.4) -> np.ndarray:
    if window_s <= 0:
        return position.copy()
    window = max(5, int(round(window_s * fps)))
    if window % 2 == 0:
        window += 1
    if len(position) <= window:
        return position.copy()
    return savgol_filter(position, window_length=window, polyorder=3, axis=0, mode="interp")


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 5 or np.std(x[finite]) <= 1e-9 or np.std(y[finite]) <= 1e-9:
        return np.nan
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def maximum_lagged_correlation(x: np.ndarray, y: np.ndarray, fps: int) -> tuple[float, float]:
    """Return the strongest speed correlation and its absolute lag within 0.5 seconds."""
    max_lag = max(1, int(round(0.5 * fps)))
    candidates: list[tuple[float, int]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = x[-lag:], y[:lag]
        elif lag > 0:
            left, right = x[:-lag], y[lag:]
        else:
            left, right = x, y
        correlation = safe_correlation(left, right)
        if np.isfinite(correlation):
            candidates.append((correlation, lag))
    if not candidates:
        return np.nan, np.nan
    correlation, lag = max(candidates, key=lambda item: item[0])
    return float(correlation), float(abs(lag) / fps)


def split_runs(indices: np.ndarray) -> list[np.ndarray]:
    if not len(indices):
        return []
    boundaries = np.flatnonzero(np.diff(indices) != 1) + 1
    return [run for run in np.split(indices, boundaries) if len(run) >= 3]


def phase_cycle_measurements(
    indices: np.ndarray,
    position_left: np.ndarray,
    position_right: np.ndarray,
    speed_left: np.ndarray,
    speed_right: np.ndarray,
    fps: int,
) -> dict[str, float] | None:
    runs = split_runs(np.sort(np.unique(indices)))
    if not runs:
        return None
    total_frames = sum(len(run) for run in runs)
    total_path = 0.0
    total_displacement = 0.0
    active_path = 0.0
    active_displacement = 0.0
    support_path = 0.0
    peak_count = 0
    slowing_values: list[tuple[float, int]] = []
    left_values = []
    right_values = []
    distance_values = []
    simultaneous = []
    lagged_correlations: list[tuple[float, float, int]] = []
    for run in runs:
        left = position_left[run]
        right = position_right[run]
        left_path = float(np.linalg.norm(np.diff(left, axis=0), axis=1).sum())
        right_path = float(np.linalg.norm(np.diff(right, axis=0), axis=1).sum())
        total_path += left_path + right_path
        total_displacement += float(np.linalg.norm(left[-1] - left[0]))
        total_displacement += float(np.linalg.norm(right[-1] - right[0]))
        if left_path >= right_path:
            active_path += left_path
            active_displacement += float(np.linalg.norm(left[-1] - left[0]))
            support_path += right_path
        else:
            active_path += right_path
            active_displacement += float(np.linalg.norm(right[-1] - right[0]))
            support_path += left_path
        left_speed = speed_left[run]
        right_speed = speed_right[run]
        combined_speed = 0.5 * (left_speed + right_speed)
        left_values.append(left_speed)
        right_values.append(right_speed)
        distance_values.append(np.linalg.norm(left - right, axis=1))
        simultaneous.append((left_speed > 10.0) & (right_speed > 10.0))
        lagged_correlation, lag_s = maximum_lagged_correlation(left_speed, right_speed, fps)
        if np.isfinite(lagged_correlation):
            lagged_correlations.append((lagged_correlation, lag_s, len(run)))
        quarter = max(1, len(run) // 4)
        slowing_values.append(
            (float(np.mean(combined_speed[:quarter]) - np.mean(combined_speed[-quarter:])), len(run))
        )
        peaks, _ = find_peaks(
            combined_speed,
            height=5.0,
            prominence=2.0,
            distance=max(1, int(round(0.25 * fps))),
        )
        peak_count += len(peaks)
    left_all = np.concatenate(left_values)
    right_all = np.concatenate(right_values)
    distance_all = np.concatenate(distance_values)
    simultaneous_all = np.concatenate(simultaneous)
    duration = total_frames / fps
    slowing = np.average(
        [value for value, _ in slowing_values],
        weights=[weight for _, weight in slowing_values],
    )
    return {
        "duration_s": float(duration),
        "combined_speed_mm_s": float(np.mean(0.5 * (left_all + right_all))),
        "path_efficiency": float(total_displacement / total_path) if total_path > 0 else np.nan,
        "active_path_efficiency": (
            float(active_displacement / active_path) if active_path > 0 else np.nan
        ),
        "support_motion_ratio": float(support_path / active_path) if active_path > 0 else np.nan,
        "bimanual_correlation": safe_correlation(left_all, right_all),
        "maximum_bimanual_correlation": (
            float(np.average([value for value, _, _ in lagged_correlations], weights=[weight for _, _, weight in lagged_correlations]))
            if lagged_correlations
            else np.nan
        ),
        "bimanual_lag_s": (
            float(np.average([lag for _, lag, _ in lagged_correlations], weights=[weight for _, _, weight in lagged_correlations]))
            if lagged_correlations
            else np.nan
        ),
        "simultaneous_motion_ratio": float(np.mean(simultaneous_all)),
        "tool_distance_mm": float(np.mean(distance_all)),
        "terminal_slowing_mm_s": float(slowing),
        "speed_peaks_per_s": float(peak_count / duration) if duration > 0 else np.nan,
    }


def coefficient_of_variation(values: pd.Series) -> float:
    values = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(values) < 2 or np.mean(values) <= 0:
        return np.nan
    return float(np.std(values, ddof=1) / np.mean(values))


def extract_features(
    phase_cache: Path,
    origin_motion: Path,
    position_window_s: float = 0.4,
    save_outputs: bool = True,
) -> pd.DataFrame:
    frames = pd.read_parquet(phase_cache / "frames.parquet")
    trials = pd.read_csv(OUT / "phase_trials_current.csv")
    cycles = pd.read_csv(OUT / "phase_cycles_current.csv")
    primary_keys = set(pd.read_csv(OUT / "phase_kinematics_by_trial.csv")["trial_key"].astype(str))
    trials = trials.copy()
    if "trial_key" not in trials:
        trials["trial_key"] = [
            phase_trial_key(dataset, trial_short)
            for dataset, trial_short in zip(trials["dataset"], trials["trial_short"])
        ]
    trial_meta = trials[trials["trial_key"].isin(primary_keys)].copy()
    cycle_context = (
        cycles[cycles["trial_key"].isin(primary_keys)]
        .groupby("trial_key")
        .agg(
            mean_phase_transitions=("n_transitions", "mean"),
            drop_events=("n_drop_events", "sum"),
        )
        .reset_index()
    )
    all_trials = pd.read_csv(OUT / "all_trial_analysis.csv")
    whole_duration = (
        all_trials[all_trials["trial_key"].isin(primary_keys)][["trial_key", "total_time"]]
        .drop_duplicates("trial_key")
        .rename(columns={"total_time": "analysed_duration_s"})
    )

    cycle_rows: list[dict[str, object]] = []
    for (dataset, trial_id, trial_short), group in frames.groupby(
        ["dataset", "trial_id", "trial_short"], sort=False
    ):
        key = phase_trial_key(str(dataset), str(trial_short))
        if key not in primary_keys:
            continue
        source = origin_motion / f"{key}.json"
        if not source.exists():
            continue
        raw = np.asarray(json.loads(source.read_text()).get("features", []), dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 14 or len(raw) < 20:
            continue
        fps = fps_for(str(dataset))
        left = smooth_positions(raw[:, 0:3], fps, window_s=position_window_s)
        right = smooth_positions(raw[:, 7:10], fps, window_s=position_window_s)
        speed_left = np.linalg.norm(np.gradient(left, 1.0 / fps, axis=0), axis=1)
        speed_right = np.linalg.norm(np.gradient(right, 1.0 / fps, axis=0), axis=1)
        ordered = group.sort_values("frame_idx")
        valid = ordered["frame_idx"].between(0, len(raw) - 1)
        ordered = ordered.loc[valid]
        for (cycle_index, phase), phase_group in ordered.groupby(
            ["cycle_index", "coarse_derived"], sort=False
        ):
            if phase not in BASE_PHASES or int(cycle_index) < 0:
                continue
            measurement = phase_cycle_measurements(
                phase_group["frame_idx"].to_numpy(int),
                left,
                right,
                speed_left,
                speed_right,
                fps,
            )
            if measurement is None:
                continue
            cycle_rows.append(
                {
                    "dataset": dataset,
                    "trial_id": trial_id,
                    "trial_key": key,
                    "cycle_index": int(cycle_index),
                    "phase": phase,
                    **measurement,
                }
            )
        for cycle_index, cycle_group in ordered.groupby("cycle_index", sort=False):
            if int(cycle_index) < 0:
                continue
            transport = cycle_group[cycle_group["coarse_derived"].isin(["transfer", "place"])]
            measurement = phase_cycle_measurements(
                transport["frame_idx"].to_numpy(int),
                left,
                right,
                speed_left,
                speed_right,
                fps,
            )
            if measurement is None:
                continue
            cycle_rows.append(
                {
                    "dataset": dataset,
                    "trial_id": trial_id,
                    "trial_key": key,
                    "cycle_index": int(cycle_index),
                    "phase": "transport_place",
                    **measurement,
                }
            )
    cycle_features = pd.DataFrame(cycle_rows)
    if save_outputs:
        cycle_features.to_csv(OUT / "phase_motion_features_by_cycle.csv", index=False)

    trial_rows: list[dict[str, object]] = []
    for key, trial_group in cycle_features.groupby("trial_key", sort=False):
        row: dict[str, object] = {"trial_key": key}
        for phase in PHASES:
            phase_group = trial_group[trial_group["phase"].eq(phase)]
            row[f"{phase}_n_cycles"] = int(phase_group["cycle_index"].nunique())
            row[f"{phase}_duration_mean_s"] = phase_group["duration_s"].mean()
            row[f"{phase}_duration_cv"] = coefficient_of_variation(phase_group["duration_s"])
            for measurement in [
                "combined_speed_mm_s",
                "path_efficiency",
                "active_path_efficiency",
                "support_motion_ratio",
                "bimanual_correlation",
                "maximum_bimanual_correlation",
                "bimanual_lag_s",
                "simultaneous_motion_ratio",
                "tool_distance_mm",
                "terminal_slowing_mm_s",
                "speed_peaks_per_s",
            ]:
                row[f"{phase}_{measurement}"] = phase_group[measurement].mean()
        trial_rows.append(row)
    features = pd.DataFrame(trial_rows)
    features = trial_meta.merge(features, on="trial_key", how="left", validate="one_to_one")
    features = features.merge(cycle_context, on="trial_key", how="left", validate="one_to_one")
    features = features.merge(whole_duration, on="trial_key", how="left", validate="one_to_one")
    if save_outputs:
        features.to_csv(OUT / "phase_motion_features_by_trial.csv", index=False)
    return features


CANDIDATE_FEATURES = {
    "grasp_duration_cv": "Grasp-duration variability",
    "transport_place_duration_cv": "Transport-duration variability",
    "reach_active_path_efficiency": "Active-tool reach efficiency",
    "transport_place_active_path_efficiency": "Active-tool transport efficiency",
    "transport_place_maximum_bimanual_correlation": "Transport speed synchrony",
    "transport_place_bimanual_lag_s": "Transport synchrony lag",
    "transport_place_support_motion_ratio": "Supporting-tool motion ratio",
    "transport_place_terminal_slowing_mm_s": "Terminal placement slowing",
    "transport_place_speed_peaks_per_s": "Transport stop-start rate",
    "mean_phase_transitions": "Phase transitions per cycle",
}


def within_cohort_standardise(values: pd.Series, cohorts: pd.Series) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for cohort in COHORT_ORDER:
        mask = cohorts.eq(cohort).to_numpy()
        x = values.to_numpy(float)[mask]
        finite = np.isfinite(x)
        if finite.sum() < 2:
            continue
        scale = np.std(x[finite], ddof=1)
        if scale <= 1e-12:
            continue
        transformed = np.full(len(x), np.nan)
        transformed[finite] = (x[finite] - np.mean(x[finite])) / scale
        output[mask] = transformed
    return output


def pooled_slope(feature: np.ndarray, experience: np.ndarray, cohorts: np.ndarray) -> float:
    finite = np.isfinite(feature) & np.isfinite(experience)
    if finite.sum() < 6:
        return np.nan
    x = experience[finite].copy()
    y = feature[finite]
    c = cohorts[finite]
    for cohort in np.unique(c):
        x[c == cohort] -= np.mean(x[c == cohort])
    denominator = np.sum(x**2)
    return float(np.sum(x * y) / denominator) if denominator > 1e-12 else np.nan


def benjamini_hochberg(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def association_analysis(features: pd.DataFrame, n_bootstrap: int = 5000, n_permutation: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    cohorts = features["dataset"].astype(str).to_numpy()
    experience = np.log10(features["total_procedures"].to_numpy(float) + 1.0)
    cohort_indices = {
        cohort: np.flatnonzero(cohorts == cohort)
        for cohort in COHORT_ORDER
    }
    rows = []
    for column, label in CANDIDATE_FEATURES.items():
        standardised = within_cohort_standardise(features[column], features["dataset"])
        observed = pooled_slope(standardised, experience, cohorts)
        boot = []
        for _ in range(n_bootstrap):
            selected = np.concatenate(
                [rng.choice(indices, size=len(indices), replace=True) for indices in cohort_indices.values()]
            )
            estimate = pooled_slope(standardised[selected], experience[selected], cohorts[selected])
            if np.isfinite(estimate):
                boot.append(estimate)
        null = []
        for _ in range(n_permutation):
            permuted = experience.copy()
            for indices in cohort_indices.values():
                permuted[indices] = rng.permutation(permuted[indices])
            estimate = pooled_slope(standardised, permuted, cohorts)
            if np.isfinite(estimate):
                null.append(estimate)
        p_value = (1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1)
        rows.append(
            {
                "feature": column,
                "label": label,
                "n_recordings": int(np.isfinite(standardised).sum()),
                "coefficient": observed,
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "permutation_p": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = benjamini_hochberg(result["permutation_p"])
    result.to_csv(OUT / "phase_motion_associations.csv", index=False)
    return result


PREDICTOR_SETS = {
    "Calibrated duration": ["analysed_duration_s"],
    "Phase timing": [
        "mean_cycle_s",
        "grasp_duration_cv",
        "transport_place_duration_cv",
        "mean_phase_transitions",
    ],
    "Phase motion": [
        "reach_active_path_efficiency",
        "transport_place_active_path_efficiency",
        "transport_place_maximum_bimanual_correlation",
        "transport_place_bimanual_lag_s",
        "transport_place_support_motion_ratio",
        "transport_place_terminal_slowing_mm_s",
        "transport_place_speed_peaks_per_s",
    ],
    "Duration + phase motion": [
        "analysed_duration_s",
        "reach_active_path_efficiency",
        "transport_place_active_path_efficiency",
        "transport_place_maximum_bimanual_correlation",
        "transport_place_bimanual_lag_s",
        "transport_place_support_motion_ratio",
        "transport_place_terminal_slowing_mm_s",
        "transport_place_speed_peaks_per_s",
    ],
}


def cohort_calibrated(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    calibrated = pd.DataFrame(index=features.index)
    for column in columns:
        output = np.full(len(features), np.nan)
        for cohort in COHORT_ORDER:
            mask = features["dataset"].eq(cohort).to_numpy()
            values = features.loc[mask, column].to_numpy(float)
            finite = np.isfinite(values)
            if not finite.any():
                continue
            median = np.nanmedian(values)
            q1, q3 = np.nanpercentile(values, [25, 75])
            scale = q3 - q1
            if scale <= 1e-12:
                scale = np.nanstd(values)
            if scale <= 1e-12 or not np.isfinite(scale):
                scale = 1.0
            output[mask] = (values - median) / scale
        calibrated[column] = output
    return calibrated


def duration_threshold(values: np.ndarray, labels: np.ndarray) -> float:
    """Choose a source-only threshold under the prespecified shorter-is-expert direction."""
    finite = np.isfinite(values)
    observed = np.sort(np.unique(values[finite]))
    if not len(observed):
        return 0.0
    candidates = np.r_[observed[0] - 1e-6, (observed[:-1] + observed[1:]) / 2, observed[-1] + 1e-6]
    scores = [
        balanced_accuracy_score(labels[finite], np.where(values[finite] <= value, "expert", "novice"))
        for value in candidates
    ]
    best = np.flatnonzero(np.isclose(scores, np.max(scores)))
    return float(candidates[best[len(best) // 2]])


def prediction_analysis(features: pd.DataFrame, n_permutation: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_columns = sorted({column for columns in PREDICTOR_SETS.values() for column in columns})
    calibrated = cohort_calibrated(features, all_columns)
    eligible = features["skill_category"].isin(["novice", "expert"]).to_numpy()
    labels = features.loc[eligible, "skill_category"].astype(str).to_numpy()
    cohorts = features.loc[eligible, "dataset"].astype(str).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    prediction_rows = []
    for name, columns in PREDICTOR_SETS.items():
        matrix = calibrated.loc[eligible, columns].reset_index(drop=True)
        observed_scores = []
        for held_out in COHORT_ORDER:
            test = cohorts == held_out
            train = ~test
            if name == "Calibrated duration":
                values = matrix.iloc[:, 0].to_numpy(float)
                threshold = duration_threshold(values[train], labels[train])
                predicted = np.where(values[test] <= threshold, "expert", "novice")
                probability = -values[test]
            else:
                pipeline = Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=0.1,
                                class_weight="balanced",
                                max_iter=10000,
                                random_state=RANDOM_SEED,
                            ),
                        ),
                    ]
                )
                pipeline.fit(matrix.loc[train], labels[train])
                predicted = pipeline.predict(matrix.loc[test])
                probability = pipeline.predict_proba(matrix.loc[test])[:, list(pipeline.classes_).index("expert")]
            ba = balanced_accuracy_score(labels[test], predicted)
            auc = roc_auc_score(labels[test] == "expert", probability)
            observed_scores.append(ba)
            rows.append(
                {
                    "model": name,
                    "held_out_cohort": held_out,
                    "n_test": int(test.sum()),
                    "balanced_accuracy": float(ba),
                    "roc_auc": float(auc),
                }
            )
            for index, prediction, score in zip(np.flatnonzero(test), predicted, probability):
                prediction_rows.append(
                    {
                        "model": name,
                        "held_out_cohort": held_out,
                        "trial_key": features.loc[eligible].iloc[index]["trial_key"],
                        "observed": labels[index],
                        "predicted": prediction,
                        "expert_probability": float(score),
                    }
                )
        observed_mean = float(np.mean(observed_scores))
        null_means = []
        for _ in range(n_permutation):
            permuted = labels.copy()
            for cohort in COHORT_ORDER:
                indices = np.flatnonzero(cohorts == cohort)
                permuted[indices] = rng.permutation(permuted[indices])
            fold_scores = []
            for held_out in COHORT_ORDER:
                test = cohorts == held_out
                train = ~test
                if name == "Calibrated duration":
                    values = matrix.iloc[:, 0].to_numpy(float)
                    threshold = duration_threshold(values[train], permuted[train])
                    predicted = np.where(values[test] <= threshold, "expert", "novice")
                else:
                    pipeline = Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                            (
                                "model",
                                LogisticRegression(
                                    C=0.1,
                                    class_weight="balanced",
                                    max_iter=10000,
                                    random_state=RANDOM_SEED,
                                ),
                            ),
                        ]
                    )
                    pipeline.fit(matrix.loc[train], permuted[train])
                    predicted = pipeline.predict(matrix.loc[test])
                fold_scores.append(balanced_accuracy_score(permuted[test], predicted))
            null_means.append(np.mean(fold_scores))
        permutation_p = (1 + np.sum(np.asarray(null_means) >= observed_mean)) / (n_permutation + 1)
        for row in rows:
            if row["model"] == name:
                row["mean_balanced_accuracy"] = observed_mean
                row["permutation_p"] = float(permutation_p)
    folds = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    folds.to_csv(OUT / "phase_motion_prediction_folds.csv", index=False)
    predictions.to_csv(OUT / "phase_motion_predictions.csv", index=False)
    return folds, predictions


def phase_pipeline(k: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_classif, k=k)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def select_feature_count(
    matrix: pd.DataFrame,
    labels: np.ndarray,
    cohorts: np.ndarray,
    held_out: str,
) -> int:
    source_cohorts = [cohort for cohort in COHORT_ORDER if cohort != held_out]
    candidates = []
    for k in range(2, matrix.shape[1] + 1):
        scores = []
        for validation_cohort in source_cohorts:
            validation = cohorts == validation_cohort
            train = (cohorts != held_out) & ~validation
            model = phase_pipeline(k)
            model.fit(matrix.loc[train], labels[train])
            scores.append(
                balanced_accuracy_score(labels[validation], model.predict(matrix.loc[validation]))
            )
        candidates.append((float(np.mean(scores)), -k, k))
    return max(candidates)[-1]


def nested_phase_prediction(
    features: pd.DataFrame,
    n_permutation: int = 2000,
    n_bootstrap: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = PREDICTOR_SETS["Phase motion"]
    calibrated = cohort_calibrated(features, columns)
    eligible = features["skill_category"].isin(["novice", "expert"]).to_numpy()
    matrix = calibrated.loc[eligible, columns].reset_index(drop=True)
    eligible_meta = features.loc[eligible].reset_index(drop=True)
    labels = eligible_meta["skill_category"].astype(str).to_numpy()
    cohorts = eligible_meta["dataset"].astype(str).to_numpy()
    folds = []
    predictions = []
    selections = []
    predicted_all = np.empty(len(labels), dtype=object)
    probability_all = np.full(len(labels), np.nan)
    for held_out in COHORT_ORDER:
        test = cohorts == held_out
        train = ~test
        selected_k = select_feature_count(matrix, labels, cohorts, held_out)
        model = phase_pipeline(selected_k)
        model.fit(matrix.loc[train], labels[train])
        predicted = model.predict(matrix.loc[test])
        probability = model.predict_proba(matrix.loc[test])[:, list(model.classes_).index("expert")]
        predicted_all[test] = predicted
        probability_all[test] = probability
        selected_columns = np.asarray(columns)[model.named_steps["select"].get_support()]
        for column in columns:
            selections.append(
                {
                    "held_out_cohort": held_out,
                    "selected_k": selected_k,
                    "feature": column,
                    "selected": column in selected_columns,
                }
            )
        folds.append(
            {
                "model": "Nested phase motion",
                "held_out_cohort": held_out,
                "n_test": int(test.sum()),
                "selected_k": selected_k,
                "balanced_accuracy": float(balanced_accuracy_score(labels[test], predicted)),
                "roc_auc": float(roc_auc_score(labels[test] == "expert", probability)),
            }
        )
        for index, prediction, score in zip(np.flatnonzero(test), predicted, probability):
            predictions.append(
                {
                    "model": "Nested phase motion",
                    "held_out_cohort": held_out,
                    "trial_key": eligible_meta.iloc[index]["trial_key"],
                    "observed": labels[index],
                    "predicted": prediction,
                    "expert_probability": float(score),
                }
            )

    observed_macro = float(np.mean([row["balanced_accuracy"] for row in folds]))
    observed_pooled = float(balanced_accuracy_score(labels, predicted_all))
    rng = np.random.default_rng(RANDOM_SEED)
    null_macro = []
    null_pooled = []
    for _ in range(n_permutation):
        permuted = labels.copy()
        for cohort in COHORT_ORDER:
            indices = np.flatnonzero(cohorts == cohort)
            permuted[indices] = rng.permutation(permuted[indices])
        permutation_predictions = np.empty(len(labels), dtype=object)
        permutation_scores = []
        for held_out in COHORT_ORDER:
            test = cohorts == held_out
            train = ~test
            selected_k = select_feature_count(matrix, permuted, cohorts, held_out)
            model = phase_pipeline(selected_k)
            model.fit(matrix.loc[train], permuted[train])
            predicted = model.predict(matrix.loc[test])
            permutation_predictions[test] = predicted
            permutation_scores.append(balanced_accuracy_score(permuted[test], predicted))
        null_macro.append(np.mean(permutation_scores))
        null_pooled.append(balanced_accuracy_score(permuted, permutation_predictions))

    cells = {
        (cohort, label): np.flatnonzero((cohorts == cohort) & (labels == label))
        for cohort in COHORT_ORDER
        for label in ["novice", "expert"]
    }
    bootstrap_macro = []
    bootstrap_pooled = []
    for _ in range(n_bootstrap):
        selected = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in cells.values()]
        )
        cohort_scores = []
        for cohort in COHORT_ORDER:
            cohort_selection = selected[cohorts[selected] == cohort]
            cohort_scores.append(
                balanced_accuracy_score(labels[cohort_selection], predicted_all[cohort_selection])
            )
        bootstrap_macro.append(np.mean(cohort_scores))
        bootstrap_pooled.append(balanced_accuracy_score(labels[selected], predicted_all[selected]))

    summary = pd.DataFrame(
        [
            {
                "macro_balanced_accuracy": observed_macro,
                "macro_ci_low": np.percentile(bootstrap_macro, 2.5),
                "macro_ci_high": np.percentile(bootstrap_macro, 97.5),
                "macro_permutation_p": (1 + np.sum(np.asarray(null_macro) >= observed_macro))
                / (n_permutation + 1),
                "pooled_balanced_accuracy": observed_pooled,
                "pooled_ci_low": np.percentile(bootstrap_pooled, 2.5),
                "pooled_ci_high": np.percentile(bootstrap_pooled, 97.5),
                "pooled_permutation_p": (1 + np.sum(np.asarray(null_pooled) >= observed_pooled))
                / (n_permutation + 1),
                "pooled_novice_sensitivity": float(
                    np.mean(predicted_all[labels == "novice"] == "novice")
                ),
                "pooled_expert_sensitivity": float(
                    np.mean(predicted_all[labels == "expert"] == "expert")
                ),
                "mean_roc_auc": float(np.mean([row["roc_auc"] for row in folds])),
            }
        ]
    )
    fold_frame = pd.DataFrame(folds)
    prediction_frame = pd.DataFrame(predictions)
    selection_frame = pd.DataFrame(selections)
    fold_frame.to_csv(OUT / "phase_motion_nested_folds.csv", index=False)
    prediction_frame.to_csv(OUT / "phase_motion_nested_predictions.csv", index=False)
    selection_frame.to_csv(OUT / "phase_motion_nested_selections.csv", index=False)
    summary.to_csv(OUT / "phase_motion_nested_summary.csv", index=False)
    return fold_frame, prediction_frame, selection_frame, summary


def nested_phase_point_estimate(features: pd.DataFrame) -> dict[str, float]:
    """Return the fully held-out point estimate without permutation or bootstrap."""
    columns = PREDICTOR_SETS["Phase motion"]
    calibrated = cohort_calibrated(features, columns)
    eligible = features["skill_category"].isin(["novice", "expert"]).to_numpy()
    matrix = calibrated.loc[eligible, columns].reset_index(drop=True)
    labels = features.loc[eligible, "skill_category"].astype(str).to_numpy()
    cohorts = features.loc[eligible, "dataset"].astype(str).to_numpy()
    predicted_all = np.empty(len(labels), dtype=object)
    cohort_balanced_accuracy = []
    cohort_auc = []
    for held_out in COHORT_ORDER:
        test = cohorts == held_out
        train = ~test
        selected_k = select_feature_count(matrix, labels, cohorts, held_out)
        model = phase_pipeline(selected_k)
        model.fit(matrix.loc[train], labels[train])
        predicted = model.predict(matrix.loc[test])
        probability = model.predict_proba(matrix.loc[test])[:, list(model.classes_).index("expert")]
        predicted_all[test] = predicted
        cohort_balanced_accuracy.append(balanced_accuracy_score(labels[test], predicted))
        cohort_auc.append(roc_auc_score(labels[test] == "expert", probability))
    return {
        "macro_balanced_accuracy": float(np.mean(cohort_balanced_accuracy)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(labels, predicted_all)),
        "mean_roc_auc": float(np.mean(cohort_auc)),
        "novice_sensitivity": float(np.mean(predicted_all[labels == "novice"] == "novice")),
        "expert_sensitivity": float(np.mean(predicted_all[labels == "expert"] == "expert")),
    }


def phase_processing_sensitivity(
    phase_cache: Path,
    origin_motion: Path,
    primary_features: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for window_s in [0.0, 0.2, 0.4, 0.8]:
        features = (
            primary_features
            if np.isclose(window_s, 0.4)
            else extract_features(
                phase_cache,
                origin_motion,
                position_window_s=window_s,
                save_outputs=False,
            )
        )
        rows.append(
            {
                "position_window_s": window_s,
                **nested_phase_point_estimate(features),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "phase_motion_processing_sensitivity.csv", index=False)
    return output


def paired_duration_contrast(
    duration_predictions: pd.DataFrame,
    phase_predictions: pd.DataFrame,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    duration = duration_predictions[
        duration_predictions["model"].eq("Calibrated duration")
    ]
    paired = duration.merge(
        phase_predictions,
        on=["held_out_cohort", "trial_key", "observed"],
        suffixes=("_duration", "_phase"),
        validate="one_to_one",
    )
    labels = paired["observed"].astype(str).to_numpy()
    cohorts = paired["held_out_cohort"].astype(str).to_numpy()
    duration_predicted = paired["predicted_duration"].astype(str).to_numpy()
    phase_predicted = paired["predicted_phase"].astype(str).to_numpy()
    rows = []
    cohort_differences = []
    for cohort in COHORT_ORDER:
        selected = cohorts == cohort
        difference = balanced_accuracy_score(
            labels[selected], phase_predicted[selected]
        ) - balanced_accuracy_score(labels[selected], duration_predicted[selected])
        cohort_differences.append(difference)
        rows.append(
            {
                "estimate": COHORT_LABELS[cohort],
                "balanced_accuracy_difference": difference,
                "ci_low": np.nan,
                "ci_high": np.nan,
            }
        )
    macro_observed = float(np.mean(cohort_differences))
    pooled_observed = float(
        balanced_accuracy_score(labels, phase_predicted)
        - balanced_accuracy_score(labels, duration_predicted)
    )
    rng = np.random.default_rng(RANDOM_SEED)
    cells = [
        np.flatnonzero((cohorts == cohort) & (labels == label))
        for cohort in COHORT_ORDER
        for label in ["novice", "expert"]
    ]
    macro_bootstrap = []
    pooled_bootstrap = []
    for _ in range(n_bootstrap):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in cells]
        )
        differences = []
        for cohort in COHORT_ORDER:
            selected = sample[cohorts[sample] == cohort]
            differences.append(
                balanced_accuracy_score(labels[selected], phase_predicted[selected])
                - balanced_accuracy_score(labels[selected], duration_predicted[selected])
            )
        macro_bootstrap.append(np.mean(differences))
        pooled_bootstrap.append(
            balanced_accuracy_score(labels[sample], phase_predicted[sample])
            - balanced_accuracy_score(labels[sample], duration_predicted[sample])
        )
    rows.extend(
        [
            {
                "estimate": "Mean across cohorts",
                "balanced_accuracy_difference": macro_observed,
                "ci_low": np.percentile(macro_bootstrap, 2.5),
                "ci_high": np.percentile(macro_bootstrap, 97.5),
            },
            {
                "estimate": "Pooled recordings",
                "balanced_accuracy_difference": pooled_observed,
                "ci_low": np.percentile(pooled_bootstrap, 2.5),
                "ci_high": np.percentile(pooled_bootstrap, 97.5),
            },
        ]
    )
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "phase_motion_duration_contrast.csv", index=False)
    return output


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "savefig.bbox": "tight",
        }
    )


def plot_results(associations: pd.DataFrame, folds: pd.DataFrame) -> None:
    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ordered = associations.sort_values("coefficient").reset_index(drop=True)
    y = np.arange(len(ordered))
    colours = np.where(ordered["q_value"] < 0.05, "#0072B2", "#666666")
    for position, row in ordered.iterrows():
        axes[0].errorbar(
            row["coefficient"],
            position,
            xerr=[[row["coefficient"] - row["ci_low"]], [row["ci_high"] - row["coefficient"]]],
            fmt="none",
            ecolor=colours[position],
            capsize=2.5,
            linewidth=1.1,
        )
    axes[0].scatter(ordered["coefficient"], y, c=colours, s=24, zorder=3)
    axes[0].axvline(0, color="#888888", linestyle=":", linewidth=0.8)
    axes[0].set_yticks(y, ordered["label"])
    axes[0].set_xlabel("Change in measurement (within-cohort SD)\nper log10 increase in procedures")
    axes[0].set_title("Phase-specific associations with procedure volume")
    axes[0].text(-0.16, 1.03, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=9)

    model_order = [
        "Calibrated duration",
        "Post hoc source-only reach share",
        "Phase timing",
        "Phase motion",
        "Nested phase motion",
    ]
    model_colours = ["#4C78A8", "#7A5195", "#E69F00", "#009E73", "#CC3311"]
    for position, (model, colour) in enumerate(zip(model_order, model_colours)):
        values = folds[folds["model"].eq(model)]
        for offset, cohort in zip([-0.10, 0.0, 0.10], COHORT_ORDER):
            score = values.loc[values["held_out_cohort"].eq(cohort), "balanced_accuracy"].iloc[0]
            axes[1].scatter(
                position + offset,
                score,
                color=COHORT_COLOURS[cohort],
                edgecolor="white",
                linewidth=0.4,
                s=27,
                zorder=3,
            )
        axes[1].scatter(
            position,
            values["balanced_accuracy"].mean(),
            marker="D",
            facecolor="white",
            edgecolor=colour,
            linewidth=1.1,
            s=38,
            zorder=4,
        )
    axes[1].axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(0.25, 1.02)
    axes[1].set_xticks(
        np.arange(len(model_order)),
        [
            "Calibrated\nduration",
            "Source-only\nreach share",
            "Phase\ntiming",
            "Phase\nmotion",
            "Nested phase\nmotion",
        ],
    )
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Novice-versus-expert cohort transfer")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=COHORT_COLOURS[cohort], label=COHORT_LABELS[cohort])
        for cohort in COHORT_ORDER
    ]
    handles.append(
        plt.Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white", markeredgecolor="#333333", label="Mean")
    )
    axes[1].legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=6.2,
    )
    axes[1].text(-0.12, 1.03, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=9)
    figure.subplots_adjust(left=0.23, right=0.99, top=0.91, bottom=0.20, wspace=0.38)
    figure.savefig(FIG / "fig28_phase_motion_predictors.png")
    figure.savefig(FIG / "fig28_phase_motion_predictors.pdf")
    plt.close(figure)


def write_table(
    associations: pd.DataFrame,
    folds: pd.DataFrame,
    selections: pd.DataFrame,
    nested_summary: pd.DataFrame,
    contrast: pd.DataFrame,
    processing: pd.DataFrame,
) -> None:
    summary = folds.groupby("model", sort=False).agg(
        mean_ba=("balanced_accuracy", "mean"),
        min_ba=("balanced_accuracy", "min"),
        max_ba=("balanced_accuracy", "max"),
        mean_auc=("roc_auc", "mean"),
        permutation_p=("permutation_p", "first"),
    )
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Representation & Mean BA $\\uparrow$ [cohort range] & Mean AUC $\\uparrow$ & Permutation $p$ $\\downarrow$ \\\\",
        "\\midrule",
    ]
    model_order = [*PREDICTOR_SETS]
    if "Post hoc source-only reach share" in summary.index:
        model_order.insert(1, "Post hoc source-only reach share")
    model_order.append("Nested phase motion")
    for model in model_order:
        row = summary.loc[model]
        lines.append(
            f"{model} & {row['mean_ba']:.2f} [{row['min_ba']:.2f}, {row['max_ba']:.2f}] & "
            f"{row['mean_auc']:.2f} & {row['permutation_p']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TAB / "phase_motion_prediction.tex").write_text("\n".join(lines) + "\n")

    association_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Measurement & $n$ & Coefficient [95\\% CI] & $p$ & $q$ \\\\",
        "\\midrule",
    ]
    for _, row in associations.sort_values("q_value").iterrows():
        association_lines.append(
            f"{row['label']} & {int(row['n_recordings'])} & {row['coefficient']:.2f} "
            f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}] & "
            f"{row['permutation_p']:.3f} & {row['q_value']:.3f} \\\\"
        )
    association_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TAB / "phase_motion_associations.tex").write_text("\n".join(association_lines) + "\n")

    selection_lines = [
        "\\begin{tabular}{lllrr}",
        "\\toprule",
        "Held-out cohort & Selected $k$ & Selected phase measurements & BA & AUC \\\\",
        "\\midrule",
    ]
    nested_folds = folds[folds["model"].eq("Nested phase motion")].set_index("held_out_cohort")
    for cohort in COHORT_ORDER:
        selected = selections[
            selections["held_out_cohort"].eq(cohort) & selections["selected"]
        ]
        labels = {
            "reach_active_path_efficiency": "reach efficiency",
            "transport_place_active_path_efficiency": "transport efficiency",
            "transport_place_maximum_bimanual_correlation": "transport synchrony",
            "transport_place_speed_peaks_per_s": "transport stop-start rate",
        }
        measurements = ", ".join(labels.get(value, value) for value in selected["feature"])
        row = nested_folds.loc[cohort]
        selection_lines.append(
            f"{COHORT_LABELS[cohort]} & {int(row['selected_k'])} & "
            f"\\parbox[t]{{5.4cm}}{{{measurements}}} & "
            f"{row['balanced_accuracy']:.2f} & {row['roc_auc']:.2f} \\\\"
        )
    selection_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TAB / "phase_motion_nested_selection.tex").write_text("\n".join(selection_lines) + "\n")

    row = nested_summary.iloc[0]
    contrast_macro = contrast[contrast["estimate"].eq("Mean across cohorts")].iloc[0]
    contrast_pooled = contrast[contrast["estimate"].eq("Pooled recordings")].iloc[0]
    summary_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Estimate & BA [95\\% CI] & Permutation $p$ & Novice sensitivity & Expert sensitivity \\\\",
        "\\midrule",
        f"Mean across cohorts & {row['macro_balanced_accuracy']:.2f} "
        f"[{row['macro_ci_low']:.2f}, {row['macro_ci_high']:.2f}] & "
        f"{row['macro_permutation_p']:.3f} & -- & -- \\\\",
        f"Pooled recordings & {row['pooled_balanced_accuracy']:.2f} "
        f"[{row['pooled_ci_low']:.2f}, {row['pooled_ci_high']:.2f}] & "
        f"{row['pooled_permutation_p']:.3f} & "
        f"{row['pooled_novice_sensitivity']:.2f} & {row['pooled_expert_sensitivity']:.2f} \\\\",
        "\\midrule",
        f"Phase motion minus duration, cohort mean & {contrast_macro['balanced_accuracy_difference']:+.2f} "
        f"[{contrast_macro['ci_low']:+.2f}, {contrast_macro['ci_high']:+.2f}] & -- & -- & -- \\\\",
        f"Phase motion minus duration, pooled & {contrast_pooled['balanced_accuracy_difference']:+.2f} "
        f"[{contrast_pooled['ci_low']:+.2f}, {contrast_pooled['ci_high']:+.2f}] & -- & -- & -- \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (TAB / "phase_motion_nested_summary.tex").write_text("\n".join(summary_lines) + "\n")

    processing_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Position window & Cohort-mean BA & Pooled BA & Mean AUC & N/E sensitivity \\\\",
        "\\midrule",
    ]
    for _, sensitivity in processing.iterrows():
        window = (
            "No smoothing"
            if np.isclose(sensitivity["position_window_s"], 0.0)
            else f"{sensitivity['position_window_s']:.1f} s"
        )
        processing_lines.append(
            f"{window} & {sensitivity['macro_balanced_accuracy']:.2f} & "
            f"{sensitivity['pooled_balanced_accuracy']:.2f} & {sensitivity['mean_roc_auc']:.2f} & "
            f"{sensitivity['novice_sensitivity']:.2f}/{sensitivity['expert_sensitivity']:.2f} \\\\"
        )
    processing_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TAB / "phase_motion_processing_sensitivity.tex").write_text(
        "\n".join(processing_lines) + "\n"
    )


def main() -> None:
    phase_cache, origin_motion = locate_inputs()
    features = extract_features(phase_cache, origin_motion)
    processing = phase_processing_sensitivity(phase_cache, origin_motion, features)
    associations = association_analysis(features)
    folds, predictions = prediction_analysis(features)
    nested_folds, nested_predictions, selections, nested_summary = nested_phase_prediction(features)
    contrast = paired_duration_contrast(predictions, nested_predictions)
    nested_folds["mean_balanced_accuracy"] = nested_summary.iloc[0]["macro_balanced_accuracy"]
    nested_folds["permutation_p"] = nested_summary.iloc[0]["macro_permutation_p"]
    display_frames = [folds, nested_folds]
    reach_path = OUT / "phase_reach_share_folds.csv"
    if reach_path.exists():
        display_frames.insert(1, pd.read_csv(reach_path))
    display_folds = pd.concat(display_frames, ignore_index=True, sort=False)
    plot_results(associations, display_folds)
    write_table(associations, display_folds, selections, nested_summary, contrast, processing)
    print(f"Phase recordings: {len(features)}")
    print("\nAssociations:")
    print(associations.sort_values("q_value").to_string(index=False))
    print("\nHeld-out-cohort prediction:")
    print(
        folds.groupby("model", sort=False)[["balanced_accuracy", "roc_auc"]]
        .agg(["mean", "min", "max"])
        .to_string()
    )
    print("\nPermutation p values:")
    print(folds.groupby("model", sort=False)["permutation_p"].first().to_string())
    print("\nNested phase-motion summary:")
    print(nested_summary.to_string(index=False))
    print("\nNested selected measurements:")
    print(selections[selections["selected"]].to_string(index=False))
    print("\nPaired contrast with calibrated duration:")
    print(contrast.to_string(index=False))
    print("\nPosition-smoothing sensitivity:")
    print(processing.to_string(index=False))


if __name__ == "__main__":
    main()
