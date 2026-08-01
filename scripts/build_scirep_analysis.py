#!/usr/bin/env python3
"""Build analysis artefacts for the Scientific Reports peg-transfer paper.

This script intentionally separates:
  1. the 115-recording dataset inventory,
  2. consistently processed kinematics for the 107 recordings whose study
     identifier occurs once, and
  3. the 38-recording phase-annotated subset for phase/cycle analyses.

The input locations can be configured with:
  LASK_CODE_ROOT    Directory containing sibling AI-ELT and BTPN-MT repos.
  LASK_PHASE_CACHE  Directory containing canonical_trials.csv and the phase
                    parquet files.
  LASK_AI_ELT_ROOT  AI-ELT project root containing motion and feature caches.
  LASK_ORIGIN_MOTION
                    Optional override for the per-recording kinematic JSON root.
  LASK_PROJECT_ROOT Optional output project root.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import hashlib
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable


def configured_path(name: str, default: Path) -> Path:
    """Return an environment-configured path with a deterministic default."""
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = configured_path("LASK_PROJECT_ROOT", SCRIPT_PROJECT_ROOT)


def default_code_root() -> Path:
    """Locate sibling analysis repositories in common checkout layouts."""
    candidates = [SCRIPT_PROJECT_ROOT.parent, *SCRIPT_PROJECT_ROOT.parents]
    for candidate in candidates:
        if (candidate / "AI-ELT").is_dir() and (candidate / "BTPN-MT").is_dir():
            return candidate
        code = candidate / "Code"
        if (code / "AI-ELT").is_dir() and (code / "BTPN-MT").is_dir():
            return code
    return SCRIPT_PROJECT_ROOT.parent


CODE_ROOT = configured_path("LASK_CODE_ROOT", default_code_root())
BTPN_CACHE = configured_path(
    "LASK_PHASE_CACHE",
    CODE_ROOT / "BTPN-MT" / "data" / "cache",
)
AI_ELT = configured_path("LASK_AI_ELT_ROOT", CODE_ROOT / "AI-ELT")
ORIGIN_MOTION = configured_path(
    "LASK_ORIGIN_MOTION",
    AI_ELT / "outputs" / "ssl" / "origin" / "origin_data" / "ORIGIN_ALL",
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "paper" / "build" / "mplconfig"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate, correlation_lags, find_peaks, savgol_filter
from scipy.spatial.transform import Rotation


OUT = ROOT / "data" / "derived"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
for d in (OUT, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def analysis_input_paths() -> dict[str, Path]:
    """Return the core files required to regenerate the reported analysis."""
    return {
        "canonical_trials": BTPN_CACHE / "canonical_trials.csv",
        "phase_trials": BTPN_CACHE / "trials.parquet",
        "phase_cycles": BTPN_CACHE / "cycles.parquet",
        "phase_frames": BTPN_CACHE / "frames.parquet",
        "combined_motion_metrics": AI_ELT
        / "outputs"
        / "motion"
        / "combined_motion_metrics.csv",
        "trimmed_extended_features": AI_ELT
        / "outputs"
        / "papers"
        / "paper1"
        / "results"
        / "extended_features_cache_trimmed.csv",
    }


def validate_analysis_inputs() -> None:
    """Fail early with actionable configuration guidance."""
    missing = [
        f"{name}: {path}"
        for name, path in analysis_input_paths().items()
        if not path.is_file()
    ]
    if not ORIGIN_MOTION.is_dir():
        missing.append(f"origin_motion_directory: {ORIGIN_MOTION}")
    if missing:
        details = "\n  ".join(missing)
        raise FileNotFoundError(
            "The analysis inputs are incomplete. Configure LASK_CODE_ROOT or "
            "LASK_PHASE_CACHE and LASK_AI_ELT_ROOT, and optionally "
            "LASK_ORIGIN_MOTION.\n  "
            f"{details}"
        )


def portable_input_path(path: Path) -> str:
    """Describe an input without writing a machine-specific absolute path."""
    try:
        return f"$LASK_PHASE_CACHE/{path.relative_to(BTPN_CACHE)}"
    except ValueError:
        pass
    try:
        return f"$LASK_AI_ELT_ROOT/{path.relative_to(AI_ELT)}"
    except ValueError:
        return path.name

DATASET_ORDER = ["BAPES2024", "6DOF2023", "7DOF2024"]
DATASET_LABELS = {
    "7DOF2024": "Urology 2 (2024)",
    "BAPES2024": "Paediatric",
    "6DOF2023": "Urology 1 (2023)",
}
SKILL_ORDER = ["novice", "intermediate", "expert"]
PHASE_ORDER = ["reach", "grasp", "transfer", "place", "nudge", "idle", "dropped"]
DATASET_COLORS = {"BAPES2024": "#0072B2", "6DOF2023": "#009E73", "7DOF2024": "#D55E00"}
BAND_ORDER = ["metric_low", "metric_middle", "metric_high"]
BAND_LABELS = {
    "metric_low": "Lower motion-score band",
    "metric_middle": "Middle motion-score band",
    "metric_high": "Upper motion-score band",
}
BAND_SHORT_LABELS = {
    "metric_low": "Lower",
    "metric_middle": "Middle",
    "metric_high": "Upper",
}
CLUSTER_COLORS = {
    "metric_low": "#009E73",
    "metric_middle": "#E69F00",
    "metric_high": "#CC3311",
}
SKILL_COLORS = {"novice": "#009E73", "intermediate": "#E69F00", "expert": "#CC3311"}
PHASE_COLORS = {
    "reach": "#4FC3F7",
    "grasp": "#66BB6A",
    "transfer": "#FFA726",
    "place": "#AB47BC",
    "nudge": "#FFD54F",
    "idle": "#B0BEC5",
    "dropped": "#E53935",
}

FEATURES = {
    "total_time": "Analysed duration",
    "bimanual_correlation": "Bimanual correlation",
    "bimanual_lag_s": "Bimanual lag",
    "bimanual_concurrent_efficiency": "Concurrent efficiency",
    "bimanual_dist_speed_corr": "Distance-speed coupling",
    "simultaneous_motion_ratio": "Simultaneous motion",
    "bimanual_symmetry": "Bimanual symmetry",
    "combined_idle_ratio": "Combined idle ratio",
    "tool1_normalized_jerk": "Left tool normalised jerk",
    "tool2_normalized_jerk": "Right tool normalised jerk",
    "tool1_active_time_ratio": "Left tool active time",
    "tool2_active_time_ratio": "Right tool active time",
    "tool1_path_length": "Left tool path length",
    "tool2_path_length": "Right tool path length",
    "tool1_avg_speed": "Left tool speed",
    "tool2_avg_speed": "Right tool speed",
    "tool_distance_cv": "Tool distance variability",
    "tool_close_proximity_ratio": "Close proximity",
    "tool1_working_volume": "Left tool working volume",
    "tool2_working_volume": "Right tool working volume",
    "tool1_working_area_xy": "Left tool working area",
    "tool2_working_area_xy": "Right tool working area",
    "tool1_range_z": "Left tool depth range",
    "tool2_range_z": "Right tool depth range",
    "tool1_rotation_per_path": "Left tool rotation per path",
    "tool2_rotation_per_path": "Right tool rotation per path",
    "tool1_angular_velocity_cv": "Left tool angular velocity variability",
    "tool2_angular_velocity_cv": "Right tool angular velocity variability",
    "tool1_num_speed_peaks": "Left tool speed peaks",
    "tool2_num_speed_peaks": "Right tool speed peaks",
    "Coordination-control composite": "Coordination and control score",
}

FEATURE_DIRECTIONS = {
    "total_time": ("$\\downarrow$", "shorter analysed task interval"),
    "bimanual_correlation": ("$\\uparrow$", "more synchronous tool motion"),
    "bimanual_lag_s": ("$\\downarrow$", "less timing delay between tools"),
    "bimanual_concurrent_efficiency": ("$\\uparrow$", "more efficient simultaneous tool use"),
    "bimanual_dist_speed_corr": ("$\\uparrow$", "more coupled spacing and speed control"),
    "simultaneous_motion_ratio": ("$\\uparrow$", "larger fraction of active frames with both tools moving"),
    "bimanual_symmetry": ("$\\uparrow$", "more balanced left/right tool use"),
    "combined_idle_ratio": ("$\\downarrow$", "less inactive task time"),
    "tool1_normalized_jerk": ("$\\downarrow$", "smoother left-tool motion"),
    "tool2_normalized_jerk": ("$\\downarrow$", "smoother right-tool motion"),
    "tool1_active_time_ratio": ("$\\uparrow$", "more active use of the left tool"),
    "tool2_active_time_ratio": ("$\\uparrow$", "more active use of the right tool"),
    "tool1_path_length": ("$\\downarrow$", "shorter left-tool travel distance"),
    "tool2_path_length": ("$\\downarrow$", "shorter right-tool travel distance"),
    "tool1_avg_speed": ("$\\uparrow$", "faster left-tool movement"),
    "tool2_avg_speed": ("$\\uparrow$", "faster right-tool movement"),
    "tool_distance_cv": ("$\\downarrow$", "steadier distance between tools"),
    "tool_close_proximity_ratio": ("$\\downarrow$", "less time with tools very close together"),
    "tool1_working_volume": ("$\\downarrow$", "more compact left-tool three-dimensional workspace"),
    "tool2_working_volume": ("$\\downarrow$", "more compact right-tool three-dimensional workspace"),
    "tool1_working_area_xy": ("$\\downarrow$", "more compact left-tool image-plane workspace"),
    "tool2_working_area_xy": ("$\\downarrow$", "more compact right-tool image-plane workspace"),
    "tool1_range_z": ("$\\downarrow$", "less left-tool camera-axis excursion"),
    "tool2_range_z": ("$\\downarrow$", "less right-tool camera-axis excursion"),
    "tool1_rotation_per_path": ("$\\downarrow$", "less left-tool rotation per millimetre travelled"),
    "tool2_rotation_per_path": ("$\\downarrow$", "less right-tool rotation per millimetre travelled"),
    "tool1_angular_velocity_cv": ("$\\downarrow$", "steadier left-tool rotational speed"),
    "tool2_angular_velocity_cv": ("$\\downarrow$", "steadier right-tool rotational speed"),
    "tool1_num_speed_peaks": ("$\\downarrow$", "fewer left-tool stop-start speed peaks"),
    "tool2_num_speed_peaks": ("$\\downarrow$", "fewer right-tool stop-start speed peaks"),
}

KMEANS_FEATURES = [
    "bimanual_correlation",
    "bimanual_lag_s",
    "simultaneous_motion_ratio",
    "bimanual_symmetry",
    "combined_idle_ratio",
    "tool_distance_cv",
    "tool1_normalized_jerk",
    "tool2_normalized_jerk",
    "tool1_active_time_ratio",
    "tool2_active_time_ratio",
    "bimanual_concurrent_efficiency",
    "tool1_working_volume",
    "tool2_working_volume",
    "tool1_working_area_xy",
    "tool2_working_area_xy",
    "tool1_range_z",
    "tool2_range_z",
    "tool1_rotation_per_path",
    "tool2_rotation_per_path",
    "tool1_angular_velocity_cv",
    "tool2_angular_velocity_cv",
    "tool_close_proximity_ratio",
]

OSATS_DOMAINS = {
    "Bimanual coordination": [
        ("bimanual_correlation", 1),
        ("simultaneous_motion_ratio", 1),
        ("bimanual_concurrent_efficiency", 1),
        ("tool_distance_cv", -1),
    ],
    "Task efficiency": [
        ("total_time", -1),
        ("combined_idle_ratio", -1),
        ("tool1_active_time_ratio", 1),
        ("tool2_active_time_ratio", 1),
        ("tool1_num_speed_peaks", -1),
        ("tool2_num_speed_peaks", -1),
    ],
    "Instrument motion control": [
        ("tool1_normalized_jerk", -1),
        ("tool2_normalized_jerk", -1),
        ("tool1_rotation_per_path", -1),
        ("tool2_rotation_per_path", -1),
        ("tool1_angular_velocity_cv", -1),
        ("tool2_angular_velocity_cv", -1),
    ],
    "Workspace excursion": [
        ("tool1_working_volume", -1),
        ("tool2_working_volume", -1),
        ("tool1_working_area_xy", -1),
        ("tool2_working_area_xy", -1),
        ("tool1_range_z", -1),
        ("tool2_range_z", -1),
    ],
}

# Mirrored tools and closely related measurements are first averaged into
# motor constructs so that duplicated channels do not receive extra weight.
OSATS_DOMAIN_CONSTRUCTS = {
    "Bimanual coordination": [
        [("bimanual_correlation", 1)],
        [("simultaneous_motion_ratio", 1), ("bimanual_concurrent_efficiency", 1)],
        [("tool_distance_cv", -1)],
    ],
    "Task efficiency": [
        [("total_time", -1)],
        [("combined_idle_ratio", -1), ("tool1_active_time_ratio", 1), ("tool2_active_time_ratio", 1)],
        [("tool1_num_speed_peaks", -1), ("tool2_num_speed_peaks", -1)],
    ],
    "Instrument motion control": [
        [("tool1_normalized_jerk", -1), ("tool2_normalized_jerk", -1)],
        [("tool1_rotation_per_path", -1), ("tool2_rotation_per_path", -1)],
        [("tool1_angular_velocity_cv", -1), ("tool2_angular_velocity_cv", -1)],
    ],
    "Workspace excursion": [
        [("tool1_working_volume", -1), ("tool2_working_volume", -1)],
        [("tool1_working_area_xy", -1), ("tool2_working_area_xy", -1)],
        [("tool1_range_z", -1), ("tool2_range_z", -1)],
    ],
}

# These two domains are available in every cohort and avoid direct timing
# measures. Task efficiency and workspace excursion are retained as
# descriptive checks that do not define the bands.
CORE_PERFORMANCE_DOMAINS = [
    "Bimanual coordination",
    "Instrument motion control",
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def nice(s: str) -> str:
    return str(s).replace("_", " ").title()


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(str(dataset), str(dataset))


def dataset_slug(dataset: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(dataset))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.03, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def image_panel_label(ax: plt.Axes, label: str, y: float = -0.08) -> None:
    """Place bracketed labels below photographic or schematic panels."""
    ax.text(
        0.5,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )


def show_center_crop(ax: plt.Axes, path: Path, target_aspect: float = 16 / 9) -> None:
    """Display an image using a centred crop without stretching it."""
    image = plt.imread(path)
    height, width = image.shape[:2]
    current_aspect = width / height
    if current_aspect > target_aspect:
        cropped_width = int(round(height * target_aspect))
        start = max(0, (width - cropped_width) // 2)
        image = image[:, start : start + cropped_width]
    elif current_aspect < target_aspect:
        cropped_height = int(round(width / target_aspect))
        start = max(0, (height - cropped_height) // 2)
        image = image[start : start + cropped_height, :]
    ax.imshow(image)
    ax.set_axis_off()


def trial_key(dataset: str, trial_number: int) -> str:
    if dataset == "6DOF2023":
        return f"6Test {int(trial_number)}"
    if dataset == "7DOF2024":
        return f"7Trial{int(trial_number)}"
    if dataset == "BAPES2024":
        return f"BTrial{int(trial_number)}"
    return f"{dataset}_{trial_number}"


def phase_trial_key(dataset: str, trial_short: str) -> str:
    text = str(trial_short)
    match = re.search(r"Trial\s*(\d+)", text)
    if match:
        return trial_key(dataset, int(match.group(1)))
    match = re.search(r"Test\s*(\d+)", text)
    if match:
        return trial_key(dataset, int(match.group(1)))
    return f"{dataset}_{text}"


def load_handedness() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paths = {
        "6DOF2023": AI_ELT / "outputs" / "ssl" / "raw" / "6DOF2023" / "ssl_results" / "trial_embeddings.json",
        "7DOF2024": AI_ELT / "outputs" / "ssl" / "raw" / "7DOF2024" / "ssl_results" / "trial_embeddings.json",
        "BAPES2024": AI_ELT / "outputs" / "ssl" / "raw" / "BAPES2024" / "ssl_results" / "trial_embeddings.json",
    }
    for dataset, path in paths.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for value in data.values():
            hand = str(value.get("handedness", "unknown")).lower().strip()
            if "right" in hand:
                compact = "right"
            elif "left" in hand:
                compact = "left"
            elif "mixed" in hand or "ambi" in hand:
                compact = "mixed"
            else:
                compact = "unknown"
            rows.append(
                {
                    "dataset": dataset,
                    "trial_number": int(value.get("trial_number", 0)),
                    "handedness": compact,
                    "procedures_last_12_months": value.get("procedures_last_12_months", np.nan),
                }
            )
    return pd.DataFrame(rows)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _savgol_positions(
    positions: np.ndarray,
    fps: int,
    window_s: float = 0.4,
) -> np.ndarray:
    """Smooth position using a short polynomial window."""
    if window_s <= 0:
        return positions.copy()
    window = max(5, int(round(window_s * fps)))
    if window % 2 == 0:
        window += 1
    if len(positions) <= window:
        return positions.copy()
    return savgol_filter(positions, window_length=window, polyorder=3, axis=0, mode="interp")


def _savgol_quaternions(
    quaternions: np.ndarray,
    fps: int,
    window_s: float = 0.0,
) -> np.ndarray:
    """Smooth sign-aligned quaternion components for sensitivity analysis."""
    q = np.asarray(quaternions, dtype=float).copy()
    if window_s <= 0 or len(q) < 3:
        return q
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(q).all() or np.any(norms <= 1e-12):
        return q
    q /= norms[:, None]
    for idx in range(1, len(q)):
        if np.dot(q[idx - 1], q[idx]) < 0:
            q[idx] *= -1
    window = max(5, int(round(window_s * fps)))
    if window % 2 == 0:
        window += 1
    if len(q) <= window:
        return q
    q = savgol_filter(q, window_length=window, polyorder=3, axis=0, mode="interp")
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms <= 1e-12):
        return np.asarray(quaternions, dtype=float).copy()
    return q / norms[:, None]


def _active_motion_bounds(
    positions1: np.ndarray,
    positions2: np.ndarray,
    fps: int,
    speed_threshold: float = 2.0,
) -> tuple[int, int]:
    """Find the first and last sustained instrument movement."""
    n = min(len(positions1), len(positions2))
    if n < 2 * fps:
        return 0, max(0, n - 1)
    dt = 1.0 / fps
    p1 = _savgol_positions(positions1[:n], fps)
    p2 = _savgol_positions(positions2[:n], fps)
    speed = np.maximum(
        np.linalg.norm(np.gradient(p1, dt, axis=0), axis=1),
        np.linalg.norm(np.gradient(p2, dt, axis=0), axis=1),
    )
    sustained_frames = max(3, int(round(0.4 * fps)))
    active = speed > speed_threshold
    runs = np.convolve(active.astype(int), np.ones(sustained_frames, dtype=int), mode="valid")
    candidates = np.flatnonzero(runs == sustained_frames)
    if not len(candidates):
        return 0, n - 1
    buffer_frames = int(round(0.5 * fps))
    start = max(0, int(candidates[0]) - buffer_frames)
    end = min(n - 1, int(candidates[-1] + sustained_frames - 1) + buffer_frames)
    return start, end


def _movement_episodes(speed: np.ndarray, fps: int, threshold: float = 5.0) -> list[tuple[int, int]]:
    moving = speed > threshold
    minimum = max(2, int(round(0.25 * fps)))
    starts = np.flatnonzero(moving & ~np.r_[False, moving[:-1]])
    ends = np.flatnonzero(moving & ~np.r_[moving[1:], False]) + 1
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= minimum]


def _idle_episodes(speed: np.ndarray, fps: int, threshold: float = 5.0) -> list[tuple[int, int]]:
    idle = speed < threshold
    minimum = max(2, int(round(0.25 * fps)))
    starts = np.flatnonzero(idle & ~np.r_[False, idle[:-1]])
    ends = np.flatnonzero(idle & ~np.r_[idle[1:], False]) + 1
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= minimum]


def _angular_metrics(quaternions_wxyz: np.ndarray, fps: int) -> tuple[float, float, float]:
    """Return total rotation, mean angular speed, and angular-speed CV."""
    q = np.asarray(quaternions_wxyz, dtype=float)
    if len(q) < 3 or q.shape[1] != 4:
        return np.nan, np.nan, np.nan
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(q).all() or np.any(norms <= 1e-12):
        return np.nan, np.nan, np.nan
    q = q / norms[:, None]
    rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    relative = rotations[:-1].inv() * rotations[1:]
    increments_deg = np.degrees(relative.magnitude())
    angular_speed = increments_deg * fps
    active = angular_speed[angular_speed > 0.1]
    total_rotation = float(np.sum(increments_deg))
    mean_speed = float(np.mean(angular_speed))
    cv = float(np.std(active, ddof=0) / np.mean(active)) if len(active) > 5 else np.nan
    return total_rotation, mean_speed, cv


def _tool_motion_features(
    positions: np.ndarray,
    quaternions: np.ndarray,
    fps: int,
    prefix: str,
    position_window_s: float = 0.4,
    orientation_window_s: float = 0.0,
) -> tuple[dict[str, float], np.ndarray]:
    dt = 1.0 / fps
    pos = _savgol_positions(
        np.asarray(positions, dtype=float),
        fps,
        window_s=position_window_s,
    )
    velocity = np.gradient(pos, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    path_length = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    duration = float(len(pos) / fps)
    elapsed = np.arange(len(pos), dtype=float) * dt
    if path_length > 0 and len(pos) > 3:
        jerk_integral = float(np.trapezoid(jerk_magnitude**2, elapsed))
        normalised_jerk = float(
            np.sqrt(0.5 * jerk_integral * ((len(pos) - 1) * dt) ** 5 / path_length**2)
        )
    else:
        normalised_jerk = np.nan
    ranges = np.ptp(pos, axis=0)
    episodes = _movement_episodes(speed, fps)
    idle_episodes = _idle_episodes(speed, fps)
    movement_durations = [(end - start) / fps for start, end in episodes]
    peaks, _ = find_peaks(
        speed,
        height=5.0,
        prominence=2.0,
        distance=max(1, int(round(0.25 * fps))),
    )
    processed_quaternions = _savgol_quaternions(
        quaternions,
        fps,
        window_s=orientation_window_s,
    )
    total_rotation, mean_angular_speed, angular_cv = _angular_metrics(
        processed_quaternions,
        fps,
    )
    features = {
        f"{prefix}_total_time": duration,
        f"{prefix}_path_length": path_length,
        f"{prefix}_path_length_3d": path_length,
        f"{prefix}_avg_speed": float(np.mean(speed)),
        f"{prefix}_max_speed": float(np.max(speed)),
        f"{prefix}_std_speed": float(np.std(speed, ddof=0)),
        f"{prefix}_speed_cv": (
            float(np.std(speed, ddof=0) / np.mean(speed)) if np.mean(speed) > 0 else np.nan
        ),
        f"{prefix}_avg_acceleration": float(np.mean(acceleration_magnitude)),
        f"{prefix}_max_acceleration": float(np.max(acceleration_magnitude)),
        f"{prefix}_std_acceleration": float(np.std(acceleration_magnitude, ddof=0)),
        f"{prefix}_accel_cv": (
            float(np.std(acceleration_magnitude, ddof=0) / np.mean(acceleration_magnitude))
            if np.mean(acceleration_magnitude) > 0
            else np.nan
        ),
        f"{prefix}_avg_jerk": float(np.mean(jerk_magnitude)),
        f"{prefix}_max_jerk": float(np.max(jerk_magnitude)),
        f"{prefix}_normalized_jerk": normalised_jerk,
        f"{prefix}_working_area_xy": float(ranges[0] * ranges[1]),
        f"{prefix}_working_volume": float(np.prod(ranges)),
        f"{prefix}_range_x": float(ranges[0]),
        f"{prefix}_range_y": float(ranges[1]),
        f"{prefix}_range_z": float(ranges[2]),
        f"{prefix}_economy_of_motion": (
            float(np.linalg.norm(pos[-1] - pos[0]) / path_length) if path_length > 0 else np.nan
        ),
        f"{prefix}_idle_time_ratio": float(np.mean(speed < 5.0)),
        f"{prefix}_active_time_ratio": float(np.mean(speed >= 5.0)),
        f"{prefix}_num_movements": float(len(episodes)),
        f"{prefix}_idle_episode_count": float(len(idle_episodes)),
        f"{prefix}_avg_movement_duration": (
            float(np.mean(movement_durations)) if movement_durations else 0.0
        ),
        f"{prefix}_num_speed_peaks": float(len(peaks)),
        f"{prefix}_total_rotation": total_rotation,
        f"{prefix}_avg_angular_velocity": mean_angular_speed,
        f"{prefix}_angular_velocity_cv": angular_cv,
        f"{prefix}_rotation_per_path": (
            float(total_rotation / path_length)
            if np.isfinite(total_rotation) and path_length > 0
            else np.nan
        ),
        f"{prefix}_rotation_economy": (
            float(total_rotation / len(episodes))
            if np.isfinite(total_rotation) and episodes
            else np.nan
        ),
    }
    return features, speed


def _trial_motion_features(
    trial_key_value: str,
    dataset: str,
    phase_bounds: dict[str, tuple[int, int]],
    position_window_s: float = 0.4,
    orientation_window_s: float = 0.0,
) -> dict[str, object] | None:
    path = ORIGIN_MOTION / f"{trial_key_value}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    raw = np.asarray(payload.get("features", []), dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 14 or len(raw) < 20:
        return None
    fps = _phase_frame_rate(dataset)
    pos1, quat1 = raw[:, 0:3], raw[:, 3:7]
    pos2, quat2 = raw[:, 7:10], raw[:, 10:14]
    if trial_key_value in phase_bounds:
        start, end = phase_bounds[trial_key_value]
        trim_method = "phase annotation"
    else:
        start, end = _active_motion_bounds(pos1, pos2, fps)
        trim_method = "sustained motion"
    start = max(0, min(int(start), len(raw) - 1))
    end = max(start, min(int(end), len(raw) - 1))
    selection = slice(start, end + 1)
    pos1, quat1 = pos1[selection], quat1[selection]
    pos2, quat2 = pos2[selection], quat2[selection]
    if len(pos1) < 20:
        return None

    tool1, speed1 = _tool_motion_features(
        pos1,
        quat1,
        fps,
        "tool1",
        position_window_s=position_window_s,
        orientation_window_s=orientation_window_s,
    )
    tool2, speed2 = _tool_motion_features(
        pos2,
        quat2,
        fps,
        "tool2",
        position_window_s=position_window_s,
        orientation_window_s=orientation_window_s,
    )
    distances = np.linalg.norm(
        _savgol_positions(pos1, fps, window_s=position_window_s)
        - _savgol_positions(pos2, fps, window_s=position_window_s),
        axis=1,
    )
    moving1_10 = speed1 > 10.0
    moving2_10 = speed2 > 10.0
    either_10 = moving1_10 | moving2_10
    both_10 = moving1_10 & moving2_10
    moving1_5 = speed1 > 5.0
    moving2_5 = speed2 > 5.0
    both_5 = moving1_5 & moving2_5
    if np.any(both_5):
        concurrent_efficiency = float(
            np.mean(
                np.minimum(speed1[both_5], speed2[both_5])
                / np.maximum(speed1[both_5], speed2[both_5])
            )
        )
    else:
        concurrent_efficiency = np.nan

    speed1_standard = (speed1 - np.mean(speed1)) / (np.std(speed1) + 1e-12)
    speed2_standard = (speed2 - np.mean(speed2)) / (np.std(speed2) + 1e-12)
    cross = correlate(speed1_standard, speed2_standard, mode="full", method="fft")
    lags = correlation_lags(len(speed1_standard), len(speed2_standard), mode="full")
    lag_limit = 5 * fps
    allowed = np.abs(lags) <= lag_limit
    best_lag_frames = int(lags[allowed][np.argmax(cross[allowed])])
    path1 = tool1["tool1_path_length"]
    path2 = tool2["tool2_path_length"]
    result: dict[str, object] = {
        "trial_key": trial_key_value,
        "trim_start_frame": start,
        "trim_end_frame": end,
        "trim_method": trim_method,
        "n_analysed_frames": int(len(pos1)),
        "fps_used": fps,
        "total_time": float(len(pos1) / fps),
        **tool1,
        **tool2,
        "bimanual_correlation": _safe_corr(speed1, speed2),
        "bimanual_lag_s": float(abs(best_lag_frames) / fps),
        "bimanual_symmetry": (
            float(min(path1, path2) / max(path1, path2)) if max(path1, path2) > 0 else np.nan
        ),
        "tool_distance_avg": float(np.mean(distances)),
        "tool_distance_std": float(np.std(distances, ddof=0)),
        "tool_distance_cv": (
            float(np.std(distances, ddof=0) / np.mean(distances))
            if np.mean(distances) > 0
            else np.nan
        ),
        "tool_close_proximity_ratio": float(np.mean(distances < 20.0)),
        "simultaneous_motion_ratio": (
            float(np.sum(both_10) / np.sum(either_10)) if np.any(either_10) else 0.0
        ),
        "bimanual_concurrent_efficiency": concurrent_efficiency,
        "bimanual_dist_speed_corr": _safe_corr(distances, speed1 + speed2),
        "combined_idle_ratio": float(
            (tool1["tool1_idle_time_ratio"] + tool2["tool2_idle_time_ratio"]) / 2
        ),
        "combined_jerk_index": float(
            (tool1["tool1_normalized_jerk"] + tool2["tool2_normalized_jerk"]) / 2
        ),
    }
    return result


def extract_primary_kinematics(
    canonical: pd.DataFrame,
    phase_frames: pd.DataFrame | None,
    position_window_s: float = 0.4,
    orientation_window_s: float = 0.0,
    output_name: str | None = "primary_kinematics.csv",
) -> pd.DataFrame:
    counts = canonical.groupby("trial_key")["trial_key"].transform("size")
    independent = canonical.loc[counts.eq(1), ["trial_key", "dataset"]].drop_duplicates()
    # Use one endpoint rule for every primary recording. Dense labels are
    # retained only to validate this automatic interval and for phase analyses.
    phase_bounds: dict[str, tuple[int, int]] = {}
    rows = []
    for row in independent.itertuples(index=False):
        features = _trial_motion_features(
            str(row.trial_key),
            str(row.dataset),
            phase_bounds,
            position_window_s=position_window_s,
            orientation_window_s=orientation_window_s,
        )
        if features is not None:
            rows.append(features)
    out = pd.DataFrame(rows)
    if output_name is not None:
        out.to_csv(OUT / output_name, index=False)
    return out


def load_all_trial_data(phase_frames: pd.DataFrame | None = None) -> pd.DataFrame:
    canon = pd.read_csv(BTPN_CACHE / "canonical_trials.csv")
    motion = pd.read_csv(AI_ELT / "outputs" / "motion" / "combined_motion_metrics.csv")
    trimmed = pd.read_csv(AI_ELT / "outputs" / "papers" / "paper1" / "results" / "extended_features_cache_trimmed.csv")
    hand = load_handedness()

    df = canon.merge(motion, on=["dataset", "trial_name"], how="left", validate="one_to_one")
    df = df.merge(trimmed, on="trial_name", how="left", validate="one_to_one", suffixes=("", "_trimmed"))
    if not hand.empty:
        df = df.merge(hand, on=["dataset", "trial_number"], how="left")
    else:
        df["handedness"] = "unknown"
    df["handedness"] = df["handedness"].fillna("unknown")
    df["cohort"] = df["dataset"].map(
        {"7DOF2024": "urology", "6DOF2023": "urology", "BAPES2024": "paediatric"}
    )
    df["cohort_label"] = df["dataset"].map(DATASET_LABELS)
    df["trial_key"] = [trial_key(ds, tn) for ds, tn in zip(df["dataset"], df["trial_number"])]
    primary = extract_primary_kinematics(df, phase_frames)
    df = df.merge(primary, on="trial_key", how="left", suffixes=("", "_primary"), validate="many_to_one")
    for col in primary.columns:
        if col == "trial_key":
            continue
        primary_col = f"{col}_primary" if col in canon.columns or col in motion.columns or col in trimmed.columns else col
        if primary_col in df.columns:
            df[col] = df[primary_col]
            if primary_col != col:
                df = df.drop(columns=primary_col)
    df["dof"] = df["dataset"].map({"7DOF2024": "7 DoF", "BAPES2024": "7 DoF", "6DOF2023": "6 DoF"})
    df["jaw"] = df["dataset"].map({"7DOF2024": "yes", "BAPES2024": "yes", "6DOF2023": "no"})
    df["fps"] = df["dataset"].map({"7DOF2024": 13, "BAPES2024": 13, "6DOF2023": 26})
    df.to_csv(OUT / "all_trial_analysis.csv", index=False)
    return df


def _phase_frame_rate(dataset: str) -> int:
    return 26 if dataset == "6DOF2023" else 13


def _trim_phase_frames_to_task(frames: pd.DataFrame) -> pd.DataFrame:
    """Retain the labelled task interval from first to last non-idle frame."""
    kept = []
    group_cols = ["dataset", "trial_id", "ann_dir"]
    for _, group in frames.sort_values(group_cols + ["frame_idx"]).groupby(group_cols, sort=False):
        active = ~group["coarse_derived"].isin(["idle", "nan", "None"])
        if not active.any():
            continue
        active_idx = np.flatnonzero(active.to_numpy())
        trimmed = group.iloc[active_idx[0] : active_idx[-1] + 1].copy()
        fps = _phase_frame_rate(str(trimmed["dataset"].iloc[0]))
        trimmed["time_s"] = np.arange(len(trimmed), dtype=float) / fps
        kept.append(trimmed)
    return pd.concat(kept, ignore_index=True) if kept else frames.iloc[0:0].copy()


def _rebuild_completed_cycles(frames: pd.DataFrame) -> pd.DataFrame:
    """Retain phase-defined cycles that reach a placement endpoint.

    A cycle is complete only when it contains at least one placement frame and
    at least one task-action frame (reach, grasp, transfer, or nudge). The
    exporter can put placement-only boundary frames into the following cycle
    index. Such a segment is attached to the immediately preceding interval
    only when that interval has no placement. Other placement-only residuals
    and attempts without placement are excluded.
    """
    rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    group_cols = ["dataset", "trial_id", "ann_dir"]
    for _, group in frames.sort_values(group_cols + ["frame_idx"]).groupby(group_cols, sort=False):
        dataset = str(group["dataset"].iloc[0])
        fps = _phase_frame_rate(dataset)
        cycle_indices = sorted(
            pd.to_numeric(group["cycle_index"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if not cycle_indices:
            continue
        cycle_frames = {
            cycle_index: group[group["cycle_index"] == cycle_index].copy()
            for cycle_index in cycle_indices
        }
        for position, cycle_index in enumerate(cycle_indices):
            cycle = cycle_frames[cycle_index]
            labels = set(cycle["coarse_derived"].astype(str))
            substantive = labels - {"idle", "place"}
            if "place" not in labels or substantive:
                continue
            previous = [
                index
                for index in cycle_indices[:position]
                if not cycle_frames[index].empty
            ]
            if previous:
                previous_index = previous[-1]
                if not cycle_frames[previous_index]["coarse_derived"].eq("place").any():
                    cycle_frames[previous_index] = (
                        pd.concat(
                            [cycle_frames[previous_index], cycle],
                            ignore_index=True,
                        )
                        .sort_values("frame_idx")
                        .reset_index(drop=True)
                    )
                    cycle_frames[cycle_index] = cycle.iloc[0:0].copy()

        for cycle_index in cycle_indices:
            cycle = cycle_frames[cycle_index]
            if cycle.empty:
                continue
            labels = cycle["coarse_derived"].astype(str)
            has_placement = bool(labels.eq("place").any())
            has_task_action = bool(
                labels.isin(["reach", "grasp", "transfer", "nudge"]).any()
            )
            if not (has_placement and has_task_action):
                excluded_rows.append(
                    {
                        "dataset": dataset,
                        "trial_id": group["trial_id"].iloc[0],
                        "ann_dir": group["ann_dir"].iloc[0],
                        "cycle_index": cycle_index,
                        "n_frames": int(len(cycle)),
                        "has_placement": has_placement,
                        "has_task_action": has_task_action,
                        "phase_sequence": ",".join(labels.drop_duplicates()),
                    }
                )
                continue
            row: dict[str, object] = {
                "dataset": dataset,
                "trial_id": group["trial_id"].iloc[0],
                "ann_dir": group["ann_dir"].iloc[0],
                "cycle_index": cycle_index,
                "duration_s": float(len(cycle) / fps),
                "dominant_phase": labels.value_counts().idxmax(),
                "n_transitions": int(labels.ne(labels.shift()).sum() - 1),
                "n_dropped_frames": int(labels.eq("dropped").sum()),
                "n_drop_events": int(
                    (labels.eq("dropped") & ~labels.shift(fill_value="").eq("dropped")).sum()
                ),
            }
            for phase in PHASE_ORDER:
                row[f"frac_{phase}"] = float(labels.eq(phase).mean())
            rows.append(row)
    pd.DataFrame(excluded_rows).to_csv(
        OUT / "phase_cycle_exclusions.csv",
        index=False,
    )
    return pd.DataFrame(rows)


def _rebuild_phase_trials(
    source_trials: pd.DataFrame,
    frames: pd.DataFrame,
    cycles: pd.DataFrame,
) -> pd.DataFrame:
    rebuilt = []
    phase_cols = [f"frac_{phase}" for phase in PHASE_ORDER]
    for (dataset, trial_id), group in frames.groupby(["dataset", "trial_id"], sort=False):
        source = source_trials[
            (source_trials["dataset"] == dataset) & (source_trials["trial_id"] == trial_id)
        ].iloc[0].to_dict()
        fps = _phase_frame_rate(str(dataset))
        labels = group["coarse_derived"].astype(str)
        trial_cycles = cycles[
            (cycles["dataset"] == dataset) & (cycles["trial_id"] == trial_id)
        ]
        durations = trial_cycles["duration_s"].to_numpy(float)
        source.update(
            {
                "n_frames": int(len(group)),
                "total_s": float(len(group) / fps),
                "n_cycles": int(len(trial_cycles)),
                "min_cycle_s": float(np.min(durations)) if len(durations) else np.nan,
                "max_cycle_s": float(np.max(durations)) if len(durations) else np.nan,
                "mean_cycle_s": float(np.mean(durations)) if len(durations) else np.nan,
                "std_cycle_s": float(np.std(durations, ddof=0)) if len(durations) else np.nan,
            }
        )
        for col in phase_cols:
            source[col] = float(labels.eq(col.replace("frac_", "")).mean())
        rebuilt.append(source)
    return pd.DataFrame(rebuilt)


def load_phase_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_trials = pd.read_parquet(BTPN_CACHE / "trials.parquet")
    frames = pd.read_parquet(BTPN_CACHE / "frames.parquet")
    meta = source_trials[["dataset", "trial_id", "trial_short", "skill_category", "total_procedures"]]
    frames = frames.merge(meta, on=["dataset", "trial_id", "trial_short"], how="left")
    frames["exported_max_cycle_index"] = frames.groupby(
        ["dataset", "trial_id", "ann_dir"]
    )["cycle_index"].transform("max")
    frames = _trim_phase_frames_to_task(frames)
    cycles = _rebuild_completed_cycles(frames)
    trials = _rebuild_phase_trials(source_trials, frames, cycles)
    cycles = cycles.merge(meta, on=["dataset", "trial_id"], how="left")
    trials["trial_key"] = [
        phase_trial_key(dataset, short)
        for dataset, short in zip(trials["dataset"], trials["trial_short"])
    ]
    cycles["trial_key"] = [
        phase_trial_key(dataset, short)
        for dataset, short in zip(cycles["dataset"], cycles["trial_short"])
    ]
    frames["trial_key"] = [
        phase_trial_key(dataset, short)
        for dataset, short in zip(frames["dataset"], frames["trial_short"])
    ]
    trials.to_csv(OUT / "phase_trials_current.csv", index=False)
    cycles.to_csv(OUT / "phase_cycles_current.csv", index=False)
    return trials, cycles, frames


def bootstrap_ci(values: Iterable[float], func=np.mean, n_boot: int = 3000) -> tuple[float, float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(20260618)
    boots = [func(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return tuple(np.percentile(boots, [2.5, 97.5]))


def bootstrap_spearman_ci(
    x_values: Iterable[float],
    y_values: Iterable[float],
    n_boot: int = 3000,
) -> tuple[float, float]:
    """Paired recording-level percentile interval for Spearman correlation."""
    pairs = pd.DataFrame({"x": x_values, "y": y_values}).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    if len(pairs) < 3:
        return (np.nan, np.nan)
    x = pairs["x"].to_numpy(float)
    y = pairs["y"].to_numpy(float)
    rng = np.random.default_rng(20260618)
    estimates = []
    for _ in range(n_boot):
        idx = rng.choice(np.arange(len(pairs)), size=len(pairs), replace=True)
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        estimates.append(stats.spearmanr(x[idx], y[idx]).statistic)
    if not estimates:
        return (np.nan, np.nan)
    return tuple(np.nanpercentile(estimates, [2.5, 97.5]))


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    y = np.asarray([v for v in b if np.isfinite(v)], dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum((v > y).sum() for v in x)
    lt = sum((v < y).sum() for v in x)
    return (gt - lt) / (len(x) * len(y))


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate correction."""
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.minimum(adjusted, 1.0)
    return q


def random_effects_correlation(
    fisher_z: Iterable[float],
    sampling_variance: Iterable[float],
) -> dict[str, float]:
    """Random-effects Fisher-z sensitivity with modified Knapp-Hartung inference."""
    z = np.asarray(list(fisher_z), dtype=float)
    variance = np.asarray(list(sampling_variance), dtype=float)
    valid = np.isfinite(z) & np.isfinite(variance) & (variance > 0)
    z = z[valid]
    variance = variance[valid]
    if len(z) < 2:
        return {
            "rho": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p": np.nan,
            "tau2": np.nan,
        }

    fixed_weights = 1.0 / variance
    fixed_mean = float(np.average(z, weights=fixed_weights))
    q_stat = float(np.sum(fixed_weights * (z - fixed_mean) ** 2))
    c_term = float(
        fixed_weights.sum()
        - np.sum(fixed_weights**2) / fixed_weights.sum()
    )
    tau2 = max(0.0, (q_stat - (len(z) - 1)) / c_term) if c_term > 0 else 0.0

    random_weights = 1.0 / (variance + tau2)
    random_mean = float(np.average(z, weights=random_weights))
    scale = float(
        np.sum(random_weights * (z - random_mean) ** 2) / (len(z) - 1)
    )
    standard_error = float(
        np.sqrt(max(1.0, scale) / random_weights.sum())
    )
    degrees_freedom = len(z) - 1
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    low_z = random_mean - critical * standard_error
    high_z = random_mean + critical * standard_error
    p_value = float(
        2 * stats.t.sf(abs(random_mean / standard_error), degrees_freedom)
    )
    return {
        "rho": float(np.tanh(random_mean)),
        "ci_low": float(np.tanh(low_z)),
        "ci_high": float(np.tanh(high_z)),
        "p": p_value,
        "tau2": float(tau2),
    }


def zscore_within_dataset(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        vals = []
        for _, g in df.groupby("dataset"):
            x = g[col].astype(float)
            sd = x.std(ddof=0)
            vals.append((x - x.mean()) / (sd if sd and np.isfinite(sd) else 1.0))
        out[col + "_z"] = pd.concat(vals).sort_index()
    return out


def feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in FEATURES
        if c in df.columns
        and pd.to_numeric(df[c], errors="coerce").dropna().nunique() > 1
    ]
    identifier_counts = df.groupby("trial_key")["trial_key"].transform("size")
    independent_df = df.loc[identifier_counts.eq(1)].copy()
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(20260618)
    for col in cols:
        cohort_deltas = []
        cohort_delta_weights = []
        cohort_kruskal_p = []
        cohort_rho_z = []
        cohort_rho_weights = []
        cohort_rho_variances = []
        for _, cohort in independent_df.groupby("dataset"):
            cohort = cohort[
                ["skill_category", "total_procedures", col]
            ].replace([np.inf, -np.inf], np.nan).dropna()
            novice = cohort.loc[
                cohort["skill_category"] == "novice", col
            ].to_numpy(float)
            expert = cohort.loc[
                cohort["skill_category"] == "expert", col
            ].to_numpy(float)
            if len(novice) and len(expert):
                cohort_deltas.append(cliffs_delta(novice, expert))
                cohort_delta_weights.append(len(novice) * len(expert))
            groups = [
                cohort.loc[cohort["skill_category"] == skill, col].to_numpy(float)
                for skill in SKILL_ORDER
            ]
            groups = [group for group in groups if len(group)]
            if len(groups) >= 2 and np.unique(np.concatenate(groups)).size > 1:
                cohort_kruskal_p.append(float(stats.kruskal(*groups).pvalue))
            if (
                len(cohort) >= 4
                and cohort[col].nunique() > 1
                and cohort["total_procedures"].nunique() > 1
            ):
                rho = float(
                    stats.spearmanr(
                        cohort[col],
                        cohort["total_procedures"],
                    ).statistic
                )
                cohort_rho_z.append(np.arctanh(np.clip(rho, -0.999999, 0.999999)))
                weight = max(len(cohort) - 3, 1)
                cohort_rho_weights.append(weight)
                cohort_rho_variances.append(1.0 / weight)

        delta = (
            float(np.average(cohort_deltas, weights=cohort_delta_weights))
            if cohort_deltas
            else np.nan
        )
        boot = []
        if cohort_deltas:
            for _ in range(3000):
                sampled_deltas = []
                sampled_weights = []
                for _, cohort in independent_df.groupby("dataset"):
                    novice = cohort.loc[
                        cohort["skill_category"] == "novice", col
                    ].dropna().to_numpy(float)
                    expert = cohort.loc[
                        cohort["skill_category"] == "expert", col
                    ].dropna().to_numpy(float)
                    if len(novice) and len(expert):
                        sampled_deltas.append(
                            cliffs_delta(
                                rng.choice(novice, size=len(novice), replace=True),
                                rng.choice(expert, size=len(expert), replace=True),
                            )
                        )
                        sampled_weights.append(len(novice) * len(expert))
                boot.append(np.average(sampled_deltas, weights=sampled_weights))
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        else:
            lo = hi = np.nan

        if cohort_kruskal_p:
            h = float(
                -2
                * np.sum(
                    np.log(np.clip(cohort_kruskal_p, np.finfo(float).tiny, 1.0))
                )
            )
            p = float(
                stats.chi2.sf(h, 2 * len(cohort_kruskal_p))
            )
        else:
            h = p = np.nan

        if cohort_rho_z:
            weights = np.asarray(cohort_rho_weights, dtype=float)
            z_values = np.asarray(cohort_rho_z, dtype=float)
            z_meta = float(np.average(z_values, weights=weights))
            z_se = float(1.0 / np.sqrt(weights.sum()))
            rho = float(np.tanh(z_meta))
            rho_p = float(2 * stats.norm.sf(abs(z_meta / z_se)))
            rho_lo, rho_hi = np.tanh(
                [z_meta - 1.96 * z_se, z_meta + 1.96 * z_se]
            )
            heterogeneity_q = float(np.sum(weights * (z_values - z_meta) ** 2))
            heterogeneity_df = max(len(z_values) - 1, 0)
            heterogeneity_i2 = (
                float(
                    max(
                        0.0,
                        100.0
                        * (heterogeneity_q - heterogeneity_df)
                        / heterogeneity_q,
                    )
                )
                if heterogeneity_q > 0 and heterogeneity_df > 0
                else 0.0
            )
        else:
            rho = rho_p = rho_lo = rho_hi = np.nan
            heterogeneity_q = heterogeneity_i2 = np.nan
        random_effects = random_effects_correlation(
            cohort_rho_z,
            cohort_rho_variances,
        )
        records.append(
            {
                "feature": col,
                "label": FEATURES[col],
                "cliffs_delta_novice_expert": delta,
                "delta_ci_low": lo,
                "delta_ci_high": hi,
                "kruskal_h": h,
                "kruskal_p": p,
                "spearman_rho_procedures": rho,
                "spearman_p": rho_p,
                "spearman_ci_low": float(rho_lo),
                "spearman_ci_high": float(rho_hi),
                "spearman_meta_i2": heterogeneity_i2,
                "spearman_random_rho": random_effects["rho"],
                "spearman_random_ci_low": random_effects["ci_low"],
                "spearman_random_ci_high": random_effects["ci_high"],
                "spearman_random_p": random_effects["p"],
                "spearman_random_tau2": random_effects["tau2"],
                "n_cohorts": int(len(cohort_rho_z)),
                "n_nonmissing": int(independent_df[col].notna().sum()),
            }
        )
    res = pd.DataFrame(records)
    res["kruskal_q"] = benjamini_hochberg(res["kruskal_p"])
    res["spearman_q"] = benjamini_hochberg(res["spearman_p"])
    res = res.sort_values("cliffs_delta_novice_expert")
    res.to_csv(OUT / "all_trial_feature_statistics.csv", index=False)
    return res


def ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, r2


def cohort_adjusted_models(df: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    """Cohort-adjusted association between experience volume and key outcomes."""
    use_df = df.copy()
    use_df["Coordination-control composite"] = (
        proxy["Coordination-control composite"].to_numpy() if len(proxy) == len(df) else np.nan
    )
    identifier_counts = use_df.groupby("trial_key")["trial_key"].transform("size")
    use_df = use_df.loc[identifier_counts.eq(1)].copy()
    outcomes = [
        "total_time",
        "bimanual_correlation",
        "tool1_avg_speed",
        "tool2_avg_speed",
        "combined_idle_ratio",
        "tool1_range_z",
        "tool2_range_z",
        "Coordination-control composite",
    ]
    rows = []
    rng = np.random.default_rng(20260618)
    for outcome in outcomes:
        if outcome not in use_df.columns:
            continue
        d = use_df[["dataset", "total_procedures", outcome]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 20:
            continue
        y_raw = d[outcome].to_numpy(float)
        y_sd = y_raw.std(ddof=0)
        if not np.isfinite(y_sd) or y_sd == 0:
            continue
        y = (y_raw - y_raw.mean()) / y_sd
        log_proc = np.log10(d["total_procedures"].to_numpy(float) + 1)
        cohort = pd.get_dummies(d["dataset"], drop_first=True).to_numpy(float)
        X_full = np.column_stack([np.ones(len(d)), log_proc, cohort])
        X_proc = np.column_stack([np.ones(len(d)), log_proc])
        X_cohort = np.column_stack([np.ones(len(d)), cohort])
        beta, r2_full = ols_fit(X_full, y)
        _, r2_proc = ols_fit(X_proc, y)
        _, r2_cohort = ols_fit(X_cohort, y)
        boot = []
        for _ in range(2000):
            idx = np.concatenate(
                [
                    rng.choice(
                        cohort_idx,
                        size=len(cohort_idx),
                        replace=True,
                    )
                    for dataset in d["dataset"].unique()
                    for cohort_idx in [
                        np.flatnonzero(d["dataset"].to_numpy() == dataset)
                    ]
                ]
            )
            b, _ = ols_fit(X_full[idx], y[idx])
            boot.append(b[1])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        permuted = []
        dataset_values = d["dataset"].to_numpy()
        for _ in range(5000):
            permuted_log_proc = log_proc.copy()
            for dataset in d["dataset"].unique():
                idx = np.flatnonzero(dataset_values == dataset)
                permuted_log_proc[idx] = rng.permutation(log_proc[idx])
            X_permuted = np.column_stack(
                [np.ones(len(d)), permuted_log_proc, cohort]
            )
            permuted.append(ols_fit(X_permuted, y)[0][1])
        procedure_p = float(
            (
                1
                + np.sum(
                    np.abs(np.asarray(permuted, dtype=float))
                    >= abs(float(beta[1]))
                )
            )
            / (len(permuted) + 1)
        )
        rows.append(
            {
                "outcome": outcome,
                "label": FEATURES.get(outcome, outcome),
                "n": int(len(d)),
                "beta_log10_procedures": float(beta[1]),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "procedure_p": procedure_p,
                "r2_procedure_only": float(r2_proc),
                "r2_cohort_only": float(r2_cohort),
                "r2_full": float(r2_full),
                "cohort_delta_r2_after_procedure": float(r2_full - r2_proc),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["procedure_q"] = benjamini_hochberg(out["procedure_p"])
    out.to_csv(OUT / "cohort_adjusted_models.csv", index=False)
    return out


def write_feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    domain_by_feature: dict[str, list[str]] = {}
    for domain, specs in OSATS_DOMAINS.items():
        for feature, _ in specs:
            domain_by_feature.setdefault(feature, []).append(domain)
    rows = []
    for feature, label in FEATURES.items():
        if feature not in df.columns:
            continue
        direction, meaning = FEATURE_DIRECTIONS.get(feature, ("", ""))
        nonmissing = int(df[feature].notna().sum())
        cohorts = [
            dataset_label(ds)
            for ds in DATASET_ORDER
            if ds in set(df.loc[df[feature].notna(), "dataset"].astype(str))
        ]
        rows.append(
            {
                "feature": feature,
                "label": label,
                "favourable_direction": direction.replace("$", "").replace("\\", ""),
                "interpretation": meaning,
                "domain": "; ".join(domain_by_feature.get(feature, [])),
                "available_cohorts": "; ".join(cohorts),
                "nonmissing_trials": nonmissing,
                "missing_trials": int(len(df) - nonmissing),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "feature_dictionary.csv", index=False)
    return out


def run_classification(df: pd.DataFrame) -> dict[str, object]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}

    all_cols = [c for c in FEATURES if c in df.columns]
    time_cols = ["total_time"] if "total_time" in df.columns else []
    identifier_counts = df.groupby("trial_key")["trial_key"].transform("size")
    use = df.loc[identifier_counts.eq(1)].dropna(
        subset=["skill_category", "dataset", "total_time"]
    ).copy()
    y = use["skill_category"].astype(str).to_numpy()
    groups = use["dataset"].astype(str).to_numpy()
    participant_groups = use["trial_key"].astype(str).to_numpy()
    out: dict[str, object] = {"available": True, "features": all_cols, "time_only_features": time_cols}

    base_models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=13),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=13),
        ),
    }
    model_specs = {
        "time_only_logistic_regression": (base_models["logistic_regression"], time_cols),
        "time_only_random_forest": (base_models["random_forest"], time_cols),
        "all_features_logistic_regression": (base_models["logistic_regression"], all_cols),
        "all_features_random_forest": (base_models["random_forest"], all_cols),
    }
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)
    cv = {}
    for name, (model, cols) in model_specs.items():
        if not cols:
            continue
        X = use[cols].replace([np.inf, -np.inf], np.nan)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=skf,
            groups=participant_groups,
            scoring="balanced_accuracy",
        )
        cv[name] = {"mean_balanced_accuracy": float(scores.mean()), "sd": float(scores.std()), "features": cols}
    out["stratified_group_5fold"] = cv

    logo = LeaveOneGroupOut()
    lodo: dict[str, list[dict[str, object]]] = {}
    for name, (model, cols) in model_specs.items():
        if not cols:
            continue
        X = use[cols].replace([np.inf, -np.inf], np.nan)
        rows = []
        for train, test in logo.split(X, y, groups):
            model.fit(X.iloc[train], y[train])
            pred = model.predict(X.iloc[test])
            rows.append(
                {
                    "held_out_dataset": str(groups[test][0]),
                    "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
                    "n_test": int(len(test)),
                }
            )
        lodo[name] = rows
    out["leave_one_dataset_out"] = lodo

    pd.DataFrame(columns=["feature", "label", "importance"]).to_csv(
        OUT / "skill_feature_importance.csv",
        index=False,
    )
    out["top_random_forest_features"] = []
    (OUT / "classification_summary.json").write_text(json.dumps(out, indent=2))
    return out


def run_metric_band_classification(clusters: pd.DataFrame) -> dict[str, object]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}

    if clusters.empty or "metric_cluster" not in clusters:
        return {"available": False, "error": "metric clusters unavailable"}
    feature_cols = [
        c
        for c in [*CORE_PERFORMANCE_DOMAINS, "Coordination-control composite"]
        if c in clusters.columns
    ]
    use = clusters.dropna(subset=["metric_cluster", "dataset", "total_procedures"]).copy()
    if not feature_cols or use["metric_cluster"].nunique() < 2:
        return {"available": False, "error": "insufficient metric bands"}

    X = use[feature_cols].replace([np.inf, -np.inf], np.nan)
    groups = use["dataset"].astype(str).to_numpy()
    participant_groups = use["trial_key"].astype(str).to_numpy()

    proc = use["total_procedures"].astype(float)
    use["procedure_fixed_band"] = np.select(
        [proc < 20, proc < 100],
        ["procedure_novice", "procedure_intermediate"],
        default="procedure_expert",
    )
    ranked_proc = proc.rank(method="first")
    use["procedure_tertile_band"] = pd.qcut(
        ranked_proc,
        q=3,
        labels=["procedure_tertile_low", "procedure_tertile_mid", "procedure_tertile_high"],
    ).astype(str)
    proc_x = np.log10(proc.to_numpy().reshape(-1, 1) + 1)
    if len(use) >= 6:
        km = KMeans(n_clusters=3, n_init=200, random_state=29)
        raw = km.fit_predict(proc_x)
        proc_tmp = pd.DataFrame({"raw": raw, "proc": proc.to_numpy()})
        ordered = proc_tmp.groupby("raw")["proc"].median().sort_values().index.to_list()
        proc_map = {
            ordered[0]: "procedure_kmeans_low",
            ordered[1]: "procedure_kmeans_mid",
            ordered[2]: "procedure_kmeans_high",
        }
        use["procedure_kmeans_band"] = [proc_map[v] for v in raw]
    else:
        use["procedure_kmeans_band"] = use["procedure_tertile_band"]

    model_factories = {
        "logistic_regression": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=29),
        ),
        "random_forest": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=29),
        ),
    }

    targets = {
        "motion_defined": {
            "label": "Motion-score bands",
            "column": "metric_cluster",
        },
        "procedure_fixed": {
            "label": "Procedure thresholds",
            "column": "procedure_fixed_band",
        },
        "procedure_tertiles": {
            "label": "Procedure tertiles",
            "column": "procedure_tertile_band",
        },
        "procedure_kmeans": {
            "label": "Procedure K-means",
            "column": "procedure_kmeans_band",
        },
    }

    results: list[dict[str, object]] = []
    details: dict[str, object] = {}
    logo = LeaveOneGroupOut()
    for target_key, target in targets.items():
        y = use[target["column"]].astype(str).to_numpy()
        class_counts = pd.Series(y).value_counts()
        if len(class_counts) < 2 or class_counts.min() < 2:
            details[target_key] = {"available": False, "class_counts": class_counts.to_dict()}
            continue
        n_splits = max(2, min(5, int(class_counts.min())))
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=29)
        details[target_key] = {
            "available": True,
            "label": target["label"],
            "class_counts": class_counts.to_dict(),
            "models": {},
        }
        for model_key, factory in model_factories.items():
            model = factory()
            scores = cross_val_score(
                model,
                X,
                y,
                cv=skf,
                groups=participant_groups,
                scoring="balanced_accuracy",
            )
            lodo_rows = []
            for train, test in logo.split(X, y, groups):
                model = factory()
                model.fit(X.iloc[train], y[train])
                pred = model.predict(X.iloc[test])
                lodo_rows.append(
                    {
                        "held_out_dataset": str(groups[test][0]),
                        "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
                        "n_test": int(len(test)),
                    }
                )
            mean_lodo = float(np.mean([r["balanced_accuracy"] for r in lodo_rows]))
            min_lodo = float(np.min([r["balanced_accuracy"] for r in lodo_rows]))
            max_lodo = float(np.max([r["balanced_accuracy"] for r in lodo_rows]))
            model_label = "Logistic regression" if model_key == "logistic_regression" else "Random forest"
            results.append(
                {
                    "target": target_key,
                    "target_label": target["label"],
                    "model": model_key,
                    "model_label": model_label,
                    "stratified_cv_balanced_accuracy": float(scores.mean()),
                    "stratified_cv_sd": float(scores.std()),
                    "leave_one_dataset_out_balanced_accuracy": mean_lodo,
                    "leave_one_dataset_out_min": min_lodo,
                    "leave_one_dataset_out_max": max_lodo,
                    "n_splits": int(n_splits),
                }
            )
            details[target_key]["models"][model_key] = {
                "stratified_cv": {
                    "mean_balanced_accuracy": float(scores.mean()),
                    "sd": float(scores.std()),
                    "n_splits": int(n_splits),
                },
                "leave_one_dataset_out": lodo_rows,
            }

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / "metric_band_classification_comparison.csv", index=False)
    legacy = {}
    if not result_df.empty:
        motion = result_df[result_df["target"] == "motion_defined"]
        legacy["stratified_cv"] = {
            f"metric_band_{r['model']}": {
                "mean_balanced_accuracy": float(r["stratified_cv_balanced_accuracy"]),
                "sd": float(r["stratified_cv_sd"]),
                "n_splits": int(r["n_splits"]),
            }
            for _, r in motion.iterrows()
        }
        legacy["leave_one_dataset_out"] = {
            f"metric_band_{model_key}": details["motion_defined"]["models"][model_key]["leave_one_dataset_out"]
            for model_key in ["logistic_regression", "random_forest"]
            if "motion_defined" in details and details["motion_defined"].get("available") and model_key in details["motion_defined"]["models"]
        }

    out = {"available": True, "features": feature_cols, "targets": details, "comparison_rows": results, **legacy}
    (OUT / "metric_band_classification_summary.json").write_text(json.dumps(out, indent=2))
    return out


def oriented_percentile_scores(df: pd.DataFrame, specs: list[tuple[str, int]]) -> pd.Series:
    parts = []
    for col, direction in specs:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce") * direction
        scored = []
        for _, g in df.assign(_x=x).groupby("dataset"):
            ranks = g["_x"].rank(pct=True)
            scored.append(ranks)
        parts.append(pd.concat(scored).sort_index())
    if not parts:
        return pd.Series(np.nan, index=df.index)
    score01 = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return 1.0 + 4.0 * score01


def construct_balanced_domain_score(
    df: pd.DataFrame,
    constructs: list[list[tuple[str, int]]],
    reference_mask: pd.Series,
) -> pd.Series:
    construct_scores = []
    for specs in constructs:
        feature_scores = []
        for col, direction in specs:
            if col not in df.columns:
                continue
            x = pd.to_numeric(df[col], errors="coerce") * direction
            ranked = pd.Series(np.nan, index=df.index, dtype=float)
            reference = df.loc[reference_mask]
            for _, idx in reference.groupby("dataset").groups.items():
                ranked.loc[idx] = x.loc[idx].rank(pct=True)
            feature_scores.append(ranked)
        if len(feature_scores) == len(specs):
            construct_scores.append(pd.concat(feature_scores, axis=1).mean(axis=1, skipna=False))
    if not construct_scores:
        return pd.Series(np.nan, index=df.index)
    score01 = pd.concat(construct_scores, axis=1).mean(axis=1, skipna=False)
    return 1.0 + 4.0 * score01


def osats_proxy_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["dataset", "cohort_label", "trial_name", "trial_number", "trial_key", "skill_category", "total_procedures"]].copy()
    identifier_counts = df.groupby("trial_key")["trial_key"].transform("size")
    reference_mask = identifier_counts.eq(1) & df["total_time"].notna()
    out["Analysed duration (s)"] = pd.to_numeric(df["total_time"], errors="coerce").where(reference_mask)
    for domain, constructs in OSATS_DOMAIN_CONSTRUCTS.items():
        out[domain] = construct_balanced_domain_score(df, constructs, reference_mask)
    domain_cols = list(OSATS_DOMAINS)
    out["Coordination-control composite"] = out[CORE_PERFORMANCE_DOMAINS].mean(
        axis=1,
        skipna=False,
    )
    out.to_csv(OUT / "osats_aligned_proxy_scores.csv", index=False)

    rows = []
    for skill, g in out.groupby("skill_category"):
        for col in domain_cols + ["Coordination-control composite"]:
            lo, hi = bootstrap_ci(g[col].dropna())
            rows.append(
                {
                    "skill_category": skill,
                    "domain": col,
                    "n": int(g[col].notna().sum()),
                    "mean_score": float(g[col].mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "osats_proxy_by_skill.csv", index=False)
    return out


def jerk_exclusion_sensitivity(
    df: pd.DataFrame,
    proxy: pd.DataFrame,
    clusters: pd.DataFrame,
    cluster_summary: dict[str, object],
) -> dict[str, object]:
    """Recalculate the score without the duration-dependent normalised-jerk construct."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    identifier_counts = df.groupby("trial_key")["trial_key"].transform("size")
    reference_mask = identifier_counts.eq(1) & df["total_time"].notna()
    no_jerk_control = construct_balanced_domain_score(
        df,
        OSATS_DOMAIN_CONSTRUCTS["Instrument motion control"][1:],
        reference_mask,
    )
    sensitivity = proxy[
        [
            "dataset",
            "trial_key",
            "Bimanual coordination",
            "Task efficiency",
            "Analysed duration (s)",
            "Coordination-control composite",
        ]
    ].copy()
    sensitivity["Motion control without normalised jerk"] = no_jerk_control
    sensitivity["Score without normalised jerk"] = sensitivity[
        ["Bimanual coordination", "Motion control without normalised jerk"]
    ].mean(axis=1, skipna=False)
    sensitivity = sensitivity.dropna(
        subset=[
            "Coordination-control composite",
            "Score without normalised jerk",
        ]
    ).merge(
        clusters[["trial_key", "metric_cluster"]].dropna(
            subset=["metric_cluster"]
        ),
        on="trial_key",
        how="inner",
        validate="one_to_one",
    )
    if sensitivity.empty:
        return {"available": False}

    band_codes = {band: i for i, band in enumerate(BAND_ORDER)}
    baseline_labels = sensitivity["metric_cluster"].map(band_codes).to_numpy(int)
    thresholds = np.asarray(
        cluster_summary.get("metric_band_thresholds", []),
        dtype=float,
    )
    if len(thresholds) != 2:
        return {"available": False}

    alternative_score = sensitivity["Score without normalised jerk"].to_numpy(float)
    fixed_labels = np.digitize(alternative_score, thresholds)
    fitted = KMeans(
        n_clusters=3,
        n_init=300,
        random_state=20260618,
    ).fit(alternative_score.reshape(-1, 1))
    centres = np.sort(fitted.cluster_centers_.ravel())
    refit_thresholds = (centres[:-1] + centres[1:]) / 2
    refit_labels = np.digitize(alternative_score, refit_thresholds)

    score_rho = float(
        stats.spearmanr(
            sensitivity["Coordination-control composite"],
            sensitivity["Score without normalised jerk"],
        ).statistic
    )
    summary_row = {
        "n": int(len(sensitivity)),
        "score_spearman_rho": score_rho,
        "median_absolute_score_change": float(
            np.median(
                np.abs(
                    sensitivity["Score without normalised jerk"]
                    - sensitivity["Coordination-control composite"]
                )
            )
        ),
        "fixed_cut_assignment_ari": float(
            adjusted_rand_score(baseline_labels, fixed_labels)
        ),
        "refit_assignment_ari": float(
            adjusted_rand_score(baseline_labels, refit_labels)
        ),
        "refit_lower_cut": float(refit_thresholds[0]),
        "refit_upper_cut": float(refit_thresholds[1]),
    }
    pd.DataFrame([summary_row]).to_csv(
        OUT / "jerk_exclusion_score_sensitivity.csv",
        index=False,
    )
    sensitivity.to_csv(OUT / "jerk_exclusion_scores.csv", index=False)

    association_rows = []
    for dataset in DATASET_ORDER:
        cohort = sensitivity[sensitivity["dataset"] == dataset]
        if cohort.empty:
            continue
        for outcome in ["Analysed duration (s)", "Task efficiency"]:
            result = stats.spearmanr(
                cohort["Score without normalised jerk"],
                cohort[outcome],
            )
            ci_low, ci_high = bootstrap_spearman_ci(
                cohort["Score without normalised jerk"],
                cohort[outcome],
            )
            association_rows.append(
                {
                    "dataset": dataset,
                    "outcome": outcome,
                    "n": int(len(cohort)),
                    "spearman_rho": float(result.statistic),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "p": float(result.pvalue),
                }
            )
    associations = pd.DataFrame(association_rows)
    associations["q"] = benjamini_hochberg(associations["p"])
    associations.to_csv(
        OUT / "jerk_exclusion_contextual_correlations.csv",
        index=False,
    )

    (TAB / "jerk_exclusion_score_sensitivity.tex").write_text(
        "\n".join(
            [
                "\\begin{tabular}{rrrrrr}",
                "\\toprule",
                "$n$ & Score $\\rho$ & Median $|\\Delta|$ & Fixed-cut ARI & Refit ARI & Refit cuts \\\\",
                "\\midrule",
                f"{summary_row['n']} & {summary_row['score_spearman_rho']:.3f} & "
                f"{summary_row['median_absolute_score_change']:.3f} & "
                f"{summary_row['fixed_cut_assignment_ari']:.3f} & "
                f"{summary_row['refit_assignment_ari']:.3f} & "
                f"{summary_row['refit_lower_cut']:.2f}, {summary_row['refit_upper_cut']:.2f} \\\\",
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
    )
    table_rows = []
    for _, row in associations.iterrows():
        q_text = "$<0.001$" if row["q"] < 0.001 else f"{row['q']:.3f}"
        table_rows.append(
            f"{dataset_label(row['dataset'])} & {row['outcome']} & {int(row['n'])} & "
            f"{row['spearman_rho']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}] & "
            f"{q_text} \\\\"
        )
    (TAB / "jerk_exclusion_contextual_correlations.tex").write_text(
        "\n".join(
            [
                "\\begin{tabular}{llrrr}",
                "\\toprule",
                "Cohort & Contextual outcome & $n$ & Spearman $\\rho$ [95\\% CI] & FDR $q$ \\\\",
                "\\midrule",
                *table_rows,
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
    )
    return {
        "available": True,
        "score_comparison": summary_row,
        "contextual_correlations": associations.to_dict(orient="records"),
    }


def processing_sensitivity(
    df: pd.DataFrame,
    phase_frames: pd.DataFrame,
    baseline_proxy: pd.DataFrame,
    baseline_clusters: pd.DataFrame,
    cluster_summary: dict[str, object],
) -> pd.DataFrame:
    """Compare the primary score under alternative position and orientation processing."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    baseline = baseline_proxy[
        ["trial_key", "Coordination-control composite"]
    ].dropna(
        subset=["Coordination-control composite"]
    ).merge(
        baseline_clusters[
            ["trial_key", "metric_cluster"]
        ].dropna(subset=["metric_cluster"]),
        on="trial_key",
        how="left",
        validate="one_to_one",
    )
    baseline = baseline.dropna(
        subset=["Coordination-control composite", "metric_cluster"]
    ).copy()
    baseline["baseline_label"] = baseline["metric_cluster"].map(
        {band: i for i, band in enumerate(BAND_ORDER)}
    )
    thresholds = np.asarray(
        cluster_summary.get("metric_band_thresholds", []),
        dtype=float,
    )
    if len(baseline) == 0 or len(thresholds) != 2:
        return pd.DataFrame()

    rows = []
    for position_window_s, orientation_window_s, label in [
        (0.0, 0.0, "No position smoothing"),
        (0.4, 0.0, "Primary processing"),
        (0.8, 0.0, "0.8 s position window"),
        (0.4, 0.4, "0.4 s orientation sensitivity"),
    ]:
        if position_window_s == 0.4 and orientation_window_s == 0.0:
            alternative = baseline[
                ["trial_key", "Coordination-control composite"]
            ].copy()
        else:
            motion = extract_primary_kinematics(
                df,
                phase_frames,
                position_window_s=position_window_s,
                orientation_window_s=orientation_window_s,
                output_name=None,
            )
            replace_cols = [c for c in motion.columns if c != "trial_key"]
            alternative_df = df.drop(columns=replace_cols, errors="ignore").merge(
                motion,
                on="trial_key",
                how="left",
                validate="many_to_one",
            )
            identifier_counts = alternative_df.groupby("trial_key")[
                "trial_key"
            ].transform("size")
            reference_mask = identifier_counts.eq(1) & alternative_df[
                "total_time"
            ].notna()
            alternative = alternative_df[["trial_key"]].copy()
            for domain in CORE_PERFORMANCE_DOMAINS:
                alternative[domain] = construct_balanced_domain_score(
                    alternative_df,
                    OSATS_DOMAIN_CONSTRUCTS[domain],
                    reference_mask,
                )
            alternative["Coordination-control composite"] = alternative[
                CORE_PERFORMANCE_DOMAINS
            ].mean(axis=1, skipna=False)

        compared = baseline.merge(
            alternative[
                ["trial_key", "Coordination-control composite"]
            ].dropna(
                subset=["Coordination-control composite"]
            ).rename(
                columns={
                    "Coordination-control composite": "alternative_score"
                }
            ),
            on="trial_key",
            how="inner",
            validate="one_to_one",
        ).dropna(subset=["alternative_score"])
        if compared.empty:
            continue
        rho = stats.spearmanr(
            compared["Coordination-control composite"],
            compared["alternative_score"],
        ).statistic
        fixed_label = np.digitize(
            compared["alternative_score"].to_numpy(float),
            thresholds,
        )
        fixed_ari = adjusted_rand_score(
            compared["baseline_label"].to_numpy(int),
            fixed_label,
        )
        values = compared["alternative_score"].to_numpy(float).reshape(-1, 1)
        fitted = KMeans(
            n_clusters=3,
            n_init=300,
            random_state=20260618,
        ).fit(values)
        centres = np.sort(fitted.cluster_centers_.ravel())
        refit_thresholds = (centres[:-1] + centres[1:]) / 2
        refit_label = np.digitize(values.ravel(), refit_thresholds)
        refit_ari = adjusted_rand_score(
            compared["baseline_label"].to_numpy(int),
            refit_label,
        )
        rows.append(
            {
                "processing": label,
                "position_window_s": position_window_s,
                "orientation_window_s": orientation_window_s,
                "n": int(len(compared)),
                "score_spearman_rho": float(rho),
                "median_absolute_score_change": float(
                    np.median(
                        np.abs(
                            compared["alternative_score"]
                            - compared["Coordination-control composite"]
                        )
                    )
                ),
                "fixed_cut_assignment_ari": float(fixed_ari),
                "refit_assignment_ari": float(refit_ari),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "processing_sensitivity.csv", index=False)
    if not out.empty:
        table_rows = [
            f"{row['processing']} & {int(row['n'])} & "
            f"{row['score_spearman_rho']:.3f} & "
            f"{row['median_absolute_score_change']:.3f} & "
            f"{row['fixed_cut_assignment_ari']:.3f} & "
            f"{row['refit_assignment_ari']:.3f} \\\\"
            for _, row in out.iterrows()
        ]
        (TAB / "processing_sensitivity.tex").write_text(
            "\n".join(
                [
                    "\\begin{tabular}{lrrrrr}",
                    "\\toprule",
                    "Processing alternative & $n$ & Score $\\rho$ & Median $|\\Delta|$ & Fixed-cut ARI & Refit ARI \\\\",
                    "\\midrule",
                    *table_rows,
                    "\\bottomrule",
                    "\\end{tabular}",
                ]
            )
        )
    return out


def kmeans_skill_clusters(df: pd.DataFrame, proxy: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        empty = df[["dataset", "trial_name", "trial_number", "skill_category", "total_procedures"]].copy()
        summary = {"available": False, "error": str(exc)}
        (OUT / "kmeans_thresholds.json").write_text(json.dumps(summary, indent=2))
        return empty, summary

    cols = [c for c in OSATS_DOMAINS if c in proxy.columns]
    core_cols = [c for c in CORE_PERFORMANCE_DOMAINS if c in proxy.columns]
    tmp = proxy[
        [
            "dataset",
            "cohort_label",
            "trial_name",
            "trial_number",
            "trial_key",
            "skill_category",
            "total_procedures",
            *cols,
            "Analysed duration (s)",
            "Coordination-control composite",
        ]
    ].copy()
    tmp["log_total_procedures"] = np.log10(tmp["total_procedures"] + 1)

    pca_cols = core_cols
    X = tmp[pca_cols].replace([np.inf, -np.inf], np.nan)
    cluster_cols = ["Coordination-control composite"]
    Xc = tmp[cluster_cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    identifier_counts = tmp.groupby("trial_key")["trial_key"].transform("size")
    fit_mask = (
        identifier_counts.eq(1).to_numpy()
        & np.isfinite(Xc).all(axis=1)
        & np.isfinite(X.to_numpy(float)).all(axis=1)
    )
    Xc_fit = Xc[fit_mask]
    tmp["metric_pc1"] = np.nan
    tmp["metric_pc2"] = np.nan
    Xs = StandardScaler().fit_transform(X.loc[fit_mask].to_numpy(float))
    pca = PCA(n_components=2, random_state=20260618)
    pcs = pca.fit_transform(Xs)
    tmp.loc[fit_mask, "metric_pc1"] = pcs[:, 0]
    tmp.loc[fit_mask, "metric_pc2"] = pcs[:, 1]
    validation_rows = []
    if len(Xc_fit) >= 6:
        for k in range(2, 7):
            km_k = KMeans(n_clusters=k, n_init=200, random_state=20260618)
            labels_k = km_k.fit_predict(Xc_fit)
            validation_rows.append(
                {
                    "k": k,
                    "silhouette": float(silhouette_score(Xc_fit, labels_k)),
                    "davies_bouldin": float(davies_bouldin_score(Xc_fit, labels_k)),
                    "calinski_harabasz": float(calinski_harabasz_score(Xc_fit, labels_k)),
                    "inertia": float(km_k.inertia_),
                }
            )

        km = KMeans(n_clusters=3, n_init=300, random_state=20260618)
        raw_cluster_fit = km.fit_predict(Xc_fit)
        metric_silhouette = float(silhouette_score(Xc_fit, raw_cluster_fit))
        metric_davies = float(davies_bouldin_score(Xc_fit, raw_cluster_fit))
        metric_calinski = float(calinski_harabasz_score(Xc_fit, raw_cluster_fit))
        ordered_centres = np.sort(km.cluster_centers_.ravel())
        thresholds = (ordered_centres[:-1] + ordered_centres[1:]) / 2
        reference_ordered = np.full(len(tmp), -1, dtype=int)
        reference_ordered[fit_mask] = np.digitize(Xc_fit.ravel(), thresholds)
        tmp["metric_cluster"] = pd.Series(np.nan, index=tmp.index, dtype=object)
        tmp.loc[fit_mask, "metric_cluster"] = [
            BAND_ORDER[i] for i in reference_ordered[fit_mask]
        ]
        rng = np.random.default_rng(20260618)
        bootstrap_thresholds = []
        bootstrap_ari = []
        for seed in range(1000):
            sample = rng.choice(
                Xc_fit.ravel(),
                size=len(Xc_fit),
                replace=True,
            ).reshape(-1, 1)
            alt = KMeans(n_clusters=3, n_init=30, random_state=seed).fit(sample)
            alt_centres = np.sort(alt.cluster_centers_.ravel())
            alt_thresholds = (alt_centres[:-1] + alt_centres[1:]) / 2
            bootstrap_thresholds.append(alt_thresholds)
            bootstrap_ari.append(
                adjusted_rand_score(
                    reference_ordered[fit_mask],
                    np.digitize(Xc_fit.ravel(), alt_thresholds),
                )
            )
        bootstrap_thresholds = np.asarray(bootstrap_thresholds)
        metric_seed_ari_mean = float(np.mean(bootstrap_ari))
        metric_seed_ari_min = float(np.min(bootstrap_ari))
        threshold_ci = np.percentile(bootstrap_thresholds, [2.5, 50, 97.5], axis=0)

        required_core_features = sorted(
            {
                col
                for domain in CORE_PERFORMANCE_DOMAINS
                for construct in OSATS_DOMAIN_CONSTRUCTS[domain]
                for col, _ in construct
                if col in df.columns
            }
        )
        complete_keys = set(
            df.loc[df[required_core_features].notna().all(axis=1), "trial_key"].astype(str)
        )
        complete_mask = (
            tmp["trial_key"].astype(str).isin(complete_keys).to_numpy()
            & fit_mask
        )
        complete_values = tmp.loc[complete_mask, ["Coordination-control composite"]].to_numpy()
        complete_model = KMeans(n_clusters=3, n_init=300, random_state=20260618).fit(complete_values)
        complete_centres = np.sort(complete_model.cluster_centers_.ravel())
        complete_thresholds = (complete_centres[:-1] + complete_centres[1:]) / 2
        complete_assignment_ari = float(
            adjusted_rand_score(
                reference_ordered[fit_mask],
                np.digitize(Xc_fit.ravel(), complete_thresholds),
            )
        )

        perturbation_ari = []
        domain_matrix = tmp[core_cols].to_numpy(float)
        for seed in range(500):
            weights = rng.uniform(0.75, 1.25, size=len(core_cols))
            perturbed_score = np.average(domain_matrix, axis=1, weights=weights)
            perturbed_model = KMeans(n_clusters=3, n_init=30, random_state=seed).fit(
                perturbed_score[fit_mask].reshape(-1, 1)
            )
            perturbed_centres = np.sort(perturbed_model.cluster_centers_.ravel())
            perturbed_thresholds = (perturbed_centres[:-1] + perturbed_centres[1:]) / 2
            perturbation_ari.append(
                adjusted_rand_score(
                    reference_ordered[fit_mask],
                    np.digitize(perturbed_score[fit_mask], perturbed_thresholds),
                )
            )

        leave_one_domain_out = []
        for omitted in core_cols:
            retained = [c for c in core_cols if c != omitted]
            score = tmp[retained].mean(axis=1).to_numpy(float)
            model = KMeans(n_clusters=3, n_init=300, random_state=20260618).fit(
                score[fit_mask].reshape(-1, 1)
            )
            centres = np.sort(model.cluster_centers_.ravel())
            cuts = (centres[:-1] + centres[1:]) / 2
            omitted_slug = re.sub(r"[^a-z0-9]+", "_", omitted.lower()).strip("_")
            omitted_clusters = pd.Series(np.nan, index=tmp.index, dtype=object)
            omitted_clusters.loc[fit_mask] = [
                BAND_ORDER[i] for i in np.digitize(score[fit_mask], cuts)
            ]
            tmp[f"metric_cluster_without_{omitted_slug}"] = omitted_clusters
            leave_one_domain_out.append(
                {
                    "omitted_domain": omitted,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(
                            reference_ordered[fit_mask],
                            np.digitize(score[fit_mask], cuts),
                        )
                    ),
                    "thresholds": [float(v) for v in cuts],
                }
            )

        leave_one_cohort_out = []
        for held_out in DATASET_ORDER:
            train_mask = (tmp["dataset"] != held_out).to_numpy() & fit_mask
            train_values = Xc[train_mask]
            model = KMeans(n_clusters=3, n_init=300, random_state=20260618).fit(train_values)
            centres = np.sort(model.cluster_centers_.ravel())
            cuts = (centres[:-1] + centres[1:]) / 2
            held_mask = (tmp["dataset"] == held_out).to_numpy() & fit_mask
            leave_one_cohort_out.append(
                {
                    "held_out_dataset": held_out,
                    "thresholds": [float(v) for v in cuts],
                    "held_out_adjusted_rand_index": float(
                        adjusted_rand_score(
                            reference_ordered[held_mask],
                            np.digitize(Xc.ravel()[held_mask], cuts),
                        )
                    ),
                }
            )
    else:
        tmp["metric_cluster"] = np.nan
        metric_silhouette = np.nan
        metric_davies = np.nan
        metric_calinski = np.nan
        metric_seed_ari_mean = np.nan
        metric_seed_ari_min = np.nan
        ordered_centres = np.array([np.nan, np.nan, np.nan])
        thresholds = np.array([np.nan, np.nan])
        threshold_ci = np.full((3, 2), np.nan)
        complete_mask = np.zeros(len(tmp), dtype=bool)
        complete_thresholds = np.array([np.nan, np.nan])
        complete_assignment_ari = np.nan
        bootstrap_ari = [np.nan]
        perturbation_ari = [np.nan]
        leave_one_domain_out = []
        leave_one_cohort_out = []
    tmp["metric_cluster_order"] = tmp["metric_cluster"].map({name: i for i, name in enumerate(BAND_ORDER)})
    tmp["data_driven_skill_band"] = tmp["metric_cluster"]
    tmp.to_csv(OUT / "kmeans_skill_clusters.csv", index=False)

    cluster_rows = []
    for cl in BAND_ORDER:
        g = tmp[tmp["metric_cluster"] == cl]
        if g.empty:
            continue
        cluster_rows.append(
            {
                "metric_cluster": cl,
                "cluster_label": BAND_LABELS[cl],
                "n": int(len(g)),
                "median_total_procedures": float(g["total_procedures"].median()),
                "procedure_iqr_low": float(g["total_procedures"].quantile(0.25)),
                "procedure_iqr_high": float(g["total_procedures"].quantile(0.75)),
                "mean_coordination_control_composite": float(
                    g["Coordination-control composite"].mean()
                ),
            }
        )
    cluster_summary = pd.DataFrame(cluster_rows)
    cluster_summary.to_csv(OUT / "kmeans_cluster_summary.csv", index=False)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUT / "kmeans_validation.csv", index=False)

    label_codes = tmp["skill_category"].map({"novice": 0, "intermediate": 1, "expert": 2})
    valid_ari = (
        label_codes.notna().to_numpy()
        & tmp["metric_cluster_order"].notna().to_numpy()
        & fit_mask
    )
    summary = {
        "available": True,
        "features": cluster_cols,
        "n_recordings_assigned": int(tmp["metric_cluster"].notna().sum()),
        "n_independent_identifiers_fitted": int(fit_mask.sum()),
        "metric_band_silhouette": metric_silhouette,
        "metric_band_davies_bouldin": metric_davies,
        "metric_band_calinski_harabasz": metric_calinski,
        "metric_band_seed_ari_mean": metric_seed_ari_mean,
        "metric_band_seed_ari_min": metric_seed_ari_min,
        "metric_band_bootstrap_assignment_ari_median": float(np.nanmedian(bootstrap_ari)),
        "metric_band_bootstrap_assignment_ari_ci": [
            float(v) for v in np.nanpercentile(bootstrap_ari, [2.5, 97.5])
        ],
        "metric_band_centres": [float(v) for v in ordered_centres],
        "metric_band_thresholds": [float(v) for v in thresholds],
        "metric_band_threshold_bootstrap_ci": [
            {
                "threshold": i + 1,
                "ci_low": float(threshold_ci[0, i]),
                "median": float(threshold_ci[1, i]),
                "ci_high": float(threshold_ci[2, i]),
            }
            for i in range(2)
        ],
        "sensitivity": {
            "complete_case": {
                "n_trials": int(complete_mask.sum()),
                "thresholds": [float(v) for v in complete_thresholds],
                "assignment_adjusted_rand_index": float(complete_assignment_ari),
            },
            "domain_weight_perturbation": {
                "n_repetitions": int(len(perturbation_ari)),
                "adjusted_rand_index_median": float(np.nanmedian(perturbation_ari)),
                "adjusted_rand_index_ci_low": float(np.nanpercentile(perturbation_ari, 2.5)),
                "adjusted_rand_index_ci_high": float(np.nanpercentile(perturbation_ari, 97.5)),
            },
            "leave_one_domain_out": leave_one_domain_out,
            "leave_one_cohort_out": leave_one_cohort_out,
        },
        "k_validation": validation.to_dict(orient="records"),
        "pca_variance": [float(v) for v in pca.explained_variance_ratio_],
        "adjusted_rand_vs_original_skill": float(
            adjusted_rand_score(
                label_codes.to_numpy()[valid_ari],
                tmp.loc[valid_ari, "metric_cluster_order"],
            )
        ),
        "cluster_summary": cluster_summary.to_dict(orient="records"),
    }
    (OUT / "kmeans_thresholds.json").write_text(json.dumps(summary, indent=2))
    return tmp, summary


def held_out_band_statistics(clusters: pd.DataFrame) -> pd.DataFrame:
    """Evaluate band gradients using outcomes that did not define the bands."""
    if clusters.empty:
        out = pd.DataFrame()
        out.to_csv(OUT / "held_out_band_statistics.csv", index=False)
        return out

    identifier_counts = clusters.groupby("trial_key")["trial_key"].transform("size")
    d = clusters.loc[identifier_counts.eq(1)].copy()
    outcomes = [
        ("Task efficiency", 1),
        ("Analysed duration (s)", -1),
        ("Workspace excursion", 1),
    ]
    rows = []
    rng = np.random.default_rng(20260618)
    for outcome, direction in outcomes:
        if outcome not in d:
            continue
        oriented = pd.Series(np.nan, index=d.index, dtype=float)
        for _, idx in d.groupby("dataset").groups.items():
            values = pd.to_numeric(d.loc[idx, outcome], errors="coerce") * direction
            oriented.loc[idx] = values.rank(pct=True)
        groups = [
            oriented.loc[d["metric_cluster"] == band].dropna().to_numpy()
            for band in BAND_ORDER
        ]
        finite_groups = [values for values in groups if len(values)]
        h, p = stats.kruskal(*finite_groups) if len(finite_groups) >= 2 else (np.nan, np.nan)
        low, middle, high = groups
        delta = cliffs_delta(high, low)
        upper_lower_p = (
            stats.mannwhitneyu(high, low, alternative="two-sided").pvalue
            if len(high) and len(low)
            else np.nan
        )
        boot = []
        if len(high) and len(low):
            for _ in range(3000):
                boot.append(
                    cliffs_delta(
                        rng.choice(high, size=len(high), replace=True),
                        rng.choice(low, size=len(low), replace=True),
                    )
                )
        lo, hi = (
            np.nanpercentile(boot, [2.5, 97.5])
            if boot
            else (np.nan, np.nan)
        )
        raw_groups = [
            pd.to_numeric(
                d.loc[d["metric_cluster"] == band, outcome],
                errors="coerce",
            ).dropna()
            for band in BAND_ORDER
        ]
        rows.append(
            {
                "outcome": outcome,
                "direction": "higher" if direction > 0 else "lower",
                "n_lower": int(len(raw_groups[0])),
                "n_middle": int(len(raw_groups[1])),
                "n_upper": int(len(raw_groups[2])),
                "median_lower": float(raw_groups[0].median()),
                "median_middle": float(raw_groups[1].median()),
                "median_upper": float(raw_groups[2].median()),
                "oriented_cliffs_delta_upper_vs_lower": float(delta),
                "delta_ci_low": float(lo),
                "delta_ci_high": float(hi),
                "kruskal_h": float(h),
                "kruskal_p": float(p),
                "upper_lower_p": float(upper_lower_p),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["kruskal_q"] = benjamini_hochberg(out["kruskal_p"])
        out["upper_lower_q"] = benjamini_hochberg(out["upper_lower_p"])
    out.to_csv(OUT / "held_out_band_statistics.csv", index=False)

    time_rows = []
    for cohort, g in [("All cohorts", d), *[(dataset_label(ds), d[d["dataset"] == ds]) for ds in DATASET_ORDER]]:
        g = g.dropna(subset=["Coordination-control composite", "Analysed duration (s)"])
        rho, p = (
            stats.spearmanr(
                g["Coordination-control composite"],
                g["Analysed duration (s)"],
            )
            if len(g) >= 3
            else (np.nan, np.nan)
        )
        time_rows.append(
            {
                "cohort": cohort,
                "n": int(len(g)),
                "spearman_rho": float(rho),
                "p": float(p),
            }
        )
    time_corr = pd.DataFrame(time_rows)
    time_corr["q"] = benjamini_hochberg(time_corr["p"])
    time_corr.to_csv(OUT / "coordination_control_time_correlation.csv", index=False)
    return out


def load_ssl_embedding_table(df: pd.DataFrame) -> pd.DataFrame:
    path = AI_ELT / "outputs" / "ssl" / "origin" / "ORIGIN_ALL" / "ssl_results" / "trial_embeddings.json"
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text())
    rows = []
    meta = df[["trial_key", "dataset", "cohort_label", "trial_number", "skill_category", "total_procedures"]]
    identifier_counts = meta.groupby("trial_key")["trial_key"].transform("size")
    meta = meta.loc[identifier_counts.eq(1)].copy()
    meta_map = meta.set_index("trial_key").to_dict(orient="index")
    for key, value in data.items():
        if key not in meta_map:
            continue
        emb = value.get("mean_embedding")
        if not isinstance(emb, list):
            continue
        row = {"trial_key": key, **meta_map[key]}
        for i, v in enumerate(emb):
            row[f"emb_{i:02d}"] = v
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        emb_cols = [c for c in out.columns if c.startswith("emb_")]
        X = StandardScaler().fit_transform(out[emb_cols].to_numpy(float))
        pca = PCA(n_components=2, random_state=20260618)
        pcs = pca.fit_transform(X)
        out["ssl_pc1"] = pcs[:, 0]
        out["ssl_pc2"] = pcs[:, 1]
        out.attrs["pca_variance"] = [float(v) for v in pca.explained_variance_ratio_]
    except Exception:
        pass
    out.to_csv(OUT / "ssl_embedding_pca.csv", index=False)
    return out


def phase_statistics(trials: pd.DataFrame, cycles: pd.DataFrame, frames: pd.DataFrame) -> dict[str, object]:
    # Segment durations in the corrected phase cache.
    seg_rows = []
    for (trial_id, ann_dir), g in frames.sort_values(["trial_id", "ann_dir", "frame_idx"]).groupby(["trial_id", "ann_dir"]):
        gg = g.reset_index(drop=True)
        if len(gg) < 2:
            continue
        dt = float(np.median(np.diff(gg["time_s"]))) if len(gg) > 2 else 0.0
        start = 0
        labels = gg["coarse_derived"].astype(str).to_numpy()
        times = gg["time_s"].to_numpy(float)
        for i in range(1, len(gg) + 1):
            if i == len(gg) or labels[i] != labels[start]:
                duration = max(0.0, (times[i - 1] - times[start]) + dt)
                seg_rows.append(
                    {
                        "dataset": gg.loc[start, "dataset"],
                        "trial_id": trial_id,
                        "skill_category": gg.loc[start, "skill_category"],
                        "phase": labels[start],
                        "duration_s": duration,
                        "n_frames": i - start,
                    }
                )
                start = i
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(OUT / "phase_segments_current.csv", index=False)
    if not seg.empty:
        seg_summary = (
            seg.groupby("phase")["duration_s"]
            .agg(n_segments="count", mean_duration_s="mean", median_duration_s="median")
            .reindex(PHASE_ORDER)
            .dropna(how="all")
            .reset_index()
        )
        seg_summary.to_csv(OUT / "phase_segment_summary.csv", index=False)

    canonical = pd.read_csv(BTPN_CACHE / "canonical_trials.csv")
    canonical["trial_key"] = [
        trial_key(dataset, trial_number)
        for dataset, trial_number in zip(canonical["dataset"], canonical["trial_number"])
    ]
    global_counts = canonical.groupby("trial_key")["trial_key"].transform("size")
    unique_keys = set(canonical.loc[global_counts.eq(1), "trial_key"])
    inferential_trials = trials[trials["trial_key"].isin(unique_keys)].copy()
    inferential_cycles = cycles[cycles["trial_key"].isin(unique_keys)].copy()

    stats_out: dict[str, object] = {}
    cycle_records = []
    trial_cycles = (
        inferential_cycles.groupby(["dataset", "skill_category", "trial_id"], as_index=False)
        .agg(mean_cycle_duration_s=("duration_s", "mean"), n_cycles=("cycle_index", "count"))
    )
    for (dataset, skill), g in trial_cycles.groupby(["dataset", "skill_category"]):
        x = g["mean_cycle_duration_s"].dropna()
        lo, hi = bootstrap_ci(x)
        cycle_records.append(
            {
                "dataset": dataset,
                "skill_category": skill,
                "n_cycles": int(g["n_cycles"].sum()),
                "n_trials": int(len(x)),
                "mean_duration_s": float(x.mean()),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    cycle_summary = pd.DataFrame(cycle_records)
    cycle_summary.to_csv(OUT / "cycle_duration_ci.csv", index=False)

    phase_records = []
    frac_cols = [c for c in inferential_trials.columns if c.startswith("frac_")]
    for skill, g in inferential_trials.groupby("skill_category"):
        for c in frac_cols:
            phase = c.replace("frac_", "")
            x = g[c].dropna()
            lo, hi = bootstrap_ci(x)
            phase_records.append(
                {
                    "skill_category": skill,
                    "phase": phase,
                    "n_trials": int(len(x)),
                    "mean_fraction": float(x.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    phase_summary = pd.DataFrame(phase_records)
    phase_summary.to_csv(OUT / "phase_fraction_ci.csv", index=False)

    stats_out["n_segments"] = int(len(seg))
    stats_out["n_inferential_phase_recordings"] = int(len(inferential_trials))
    stats_out["n_unique_phase_identifiers"] = int(inferential_trials["trial_key"].nunique())
    stats_out["mean_cycle_duration_by_skill"] = (
        trial_cycles.groupby("skill_category")["mean_cycle_duration_s"].mean().to_dict()
    )
    return stats_out


def phase_specific_kinematics(
    frames: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Exploratory phase-specific kinematics for phase-labelled primary recordings."""
    base = AI_ELT / "outputs" / "ssl" / "origin" / "origin_data" / "ORIGIN_ALL"
    if not base.exists():
        out = pd.DataFrame()
        out.to_csv(OUT / "phase_kinematics_summary.csv", index=False)
        return out
    rows = []
    for (dataset, trial_id, trial_short), g in frames.groupby(["dataset", "trial_id", "trial_short"]):
        key = phase_trial_key(dataset, trial_short)
        path = base / f"{key}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        feats = np.asarray(data.get("features", []), dtype=float)
        if feats.ndim != 2 or feats.shape[1] < 14 or len(feats) < 3:
            continue
        phase_g = g.sort_values("frame_idx").reset_index(drop=True)
        phase_frame_idx = phase_g["frame_idx"].to_numpy(int)
        phase_labels = phase_g["coarse_derived"].astype(str).to_numpy()
        if len(phase_frame_idx) < 3:
            continue
        valid_frames = (phase_frame_idx >= 0) & (phase_frame_idx < len(feats))
        phase_frame_idx = phase_frame_idx[valid_frames]
        labels = phase_labels[valid_frames]
        if len(phase_frame_idx) < 3:
            continue
        fps = _phase_frame_rate(str(dataset))
        p1 = _savgol_positions(feats[:, 0:3], fps)[phase_frame_idx]
        p2 = _savgol_positions(feats[:, 7:10], fps)[phase_frame_idx]
        frame_steps = np.diff(phase_frame_idx)
        dt = frame_steps / fps
        valid = frame_steps > 0
        d1 = np.linalg.norm(np.diff(p1, axis=0), axis=1)
        d2 = np.linalg.norm(np.diff(p2, axis=0), axis=1)
        interval_phase = labels[1:]
        for phase in PHASE_ORDER:
            mask = valid & (interval_phase == phase)
            if not mask.any():
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "trial_id": trial_id,
                    "trial_key": key,
                    "phase": phase,
                    "duration_s": float(dt[mask].sum()),
                    "tool1_path_mm": float(d1[mask].sum()),
                    "tool2_path_mm": float(d2[mask].sum()),
                    "tool1_mean_speed_mm_s": float(d1[mask].sum() / dt[mask].sum()),
                    "tool2_mean_speed_mm_s": float(d2[mask].sum() / dt[mask].sum()),
                    "combined_path_rate_mm_s": float((d1[mask].sum() + d2[mask].sum()) / dt[mask].sum()),
                    "n_intervals": int(mask.sum()),
                }
            )
    phase_df = pd.DataFrame(rows)
    if not phase_df.empty:
        valid_keys = set(clusters.loc[clusters["metric_cluster"].notna(), "trial_key"])
        phase_df = phase_df[phase_df["trial_key"].isin(valid_keys)].copy()
    phase_df.to_csv(OUT / "phase_kinematics_by_trial.csv", index=False)
    if phase_df.empty:
        out = pd.DataFrame()
    else:
        out = (
            phase_df.groupby("phase")
            .agg(
                n_trials=("trial_key", "nunique"),
                duration_s=("duration_s", "sum"),
                tool1_speed_mm_s=("tool1_mean_speed_mm_s", "mean"),
                tool2_speed_mm_s=("tool2_mean_speed_mm_s", "mean"),
                combined_path_rate_mm_s=("combined_path_rate_mm_s", "mean"),
            )
            .reindex(PHASE_ORDER)
            .dropna(how="all")
            .reset_index()
        )
    out.to_csv(OUT / "phase_kinematics_summary.csv", index=False)
    return out


def trimming_rule_validation(
    frames: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Compare automatic motion bounds with dense non-idle annotation bounds."""
    valid_keys = set(
        clusters.loc[clusters["metric_cluster"].notna(), "trial_key"].astype(str)
    )
    rows = []
    for (dataset, key), group in frames.groupby(["dataset", "trial_key"]):
        key = str(key)
        if key not in valid_keys:
            continue
        path = ORIGIN_MOTION / f"{key}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        raw = np.asarray(payload.get("features", []), dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 14 or len(raw) < 20:
            continue
        fps = _phase_frame_rate(str(dataset))
        manual_start = int(group["frame_idx"].min())
        manual_end = int(group["frame_idx"].max())
        automatic_start, automatic_end = _active_motion_bounds(
            raw[:, 0:3],
            raw[:, 7:10],
            fps,
        )
        manual_start = max(0, min(manual_start, len(raw) - 1))
        manual_end = max(manual_start, min(manual_end, len(raw) - 1))
        overlap = max(
            0,
            min(manual_end, automatic_end)
            - max(manual_start, automatic_start)
            + 1,
        )
        union = (
            max(manual_end, automatic_end)
            - min(manual_start, automatic_start)
            + 1
        )
        manual_duration = (manual_end - manual_start + 1) / fps
        automatic_duration = (automatic_end - automatic_start + 1) / fps
        rows.append(
            {
                "dataset": str(dataset),
                "trial_key": key,
                "manual_start_frame": manual_start,
                "manual_end_frame": manual_end,
                "automatic_start_frame": automatic_start,
                "automatic_end_frame": automatic_end,
                "start_error_s": (automatic_start - manual_start) / fps,
                "end_error_s": (automatic_end - manual_end) / fps,
                "duration_error_s": automatic_duration - manual_duration,
                "absolute_duration_error_s": abs(
                    automatic_duration - manual_duration
                ),
                "interval_iou": overlap / union if union > 0 else np.nan,
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "trimming_rule_validation_by_recording.csv", index=False)
    if detail.empty:
        return detail
    summary_rows = []
    for label, group in [
        ("All cohorts", detail),
        *[
            (dataset_label(dataset), detail[detail["dataset"] == dataset])
            for dataset in DATASET_ORDER
        ],
    ]:
        if group.empty:
            continue
        summary_rows.append(
            {
                "cohort": label,
                "n": int(len(group)),
                "median_abs_start_error_s": float(
                    np.median(np.abs(group["start_error_s"]))
                ),
                "median_abs_end_error_s": float(
                    np.median(np.abs(group["end_error_s"]))
                ),
                "median_abs_duration_error_s": float(
                    np.median(group["absolute_duration_error_s"])
                ),
                "median_interval_iou": float(
                    np.median(group["interval_iou"])
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "trimming_rule_validation.csv", index=False)
    table_rows = [
        f"{row['cohort']} & {int(row['n'])} & "
        f"{row['median_abs_start_error_s']:.2f} & "
        f"{row['median_abs_end_error_s']:.2f} & "
        f"{row['median_abs_duration_error_s']:.2f} & "
        f"{row['median_interval_iou']:.3f} \\\\"
        for _, row in summary.iterrows()
    ]
    (TAB / "trimming_rule_validation.tex").write_text(
        "\n".join(
            [
                "\\begin{tabular}{lrrrrr}",
                "\\toprule",
                "Cohort & $n$ & Start error (s) & End error (s) & Duration error (s) & Interval overlap \\\\",
                "\\midrule",
                *table_rows,
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
    )
    return summary


def load_origin_motion(df: pd.DataFrame, max_points_per_trial: int = 350) -> pd.DataFrame:
    base = AI_ELT / "outputs" / "ssl" / "origin" / "origin_data" / "ORIGIN_ALL"
    if not base.exists():
        return pd.DataFrame()
    meta = df[["trial_key", "dataset", "cohort_label", "trial_number", "skill_category", "total_procedures"]]
    identifier_counts = meta.groupby("trial_key")["trial_key"].transform("size")
    meta = (
        meta.loc[identifier_counts.eq(1)]
        .set_index("trial_key")
        .to_dict(orient="index")
    )
    rows = []
    rng = np.random.default_rng(20260618)
    for path in sorted(base.glob("*.json")):
        if path.name == "metadata.json":
            continue
        data = json.loads(path.read_text())
        key = data.get("name")
        if key not in meta:
            continue
        feats = np.asarray(data.get("features", []), dtype=float)
        times = np.asarray(data.get("times", []), dtype=float)
        if feats.ndim != 2 or feats.shape[1] < 14:
            continue
        n = len(feats)
        if n > max_points_per_trial:
            idx = np.sort(rng.choice(np.arange(n), size=max_points_per_trial, replace=False))
        else:
            idx = np.arange(n)
        for i in idx:
            row = {
                "trial_key": key,
                "frame_sample": int(i),
                "time_s": float(times[i]) if i < len(times) else np.nan,
                "tool1_x": feats[i, 0],
                "tool1_y": feats[i, 1],
                "tool1_z": feats[i, 2],
                "tool1_qw": feats[i, 3],
                "tool1_qx": feats[i, 4],
                "tool1_qy": feats[i, 5],
                "tool1_qz": feats[i, 6],
                "tool2_x": feats[i, 7],
                "tool2_y": feats[i, 8],
                "tool2_z": feats[i, 9],
                "tool2_qw": feats[i, 10],
                "tool2_qx": feats[i, 11],
                "tool2_qy": feats[i, 12],
                "tool2_qz": feats[i, 13],
                **meta[key],
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(OUT / "origin_motion_sample.csv", index=False)
    return out


def plot_cohort(df: pd.DataFrame, phase_trials: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    counts = pd.crosstab(df["dataset"], df["skill_category"]).reindex(DATASET_ORDER).fillna(0)
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))
    for skill in SKILL_ORDER:
        vals = counts.get(skill, pd.Series(0, index=counts.index)).to_numpy()
        ax.bar(x, vals, bottom=bottom, color=SKILL_COLORS[skill], label=nice(skill), width=0.7)
        bottom += vals
    ax.set_xticks(x, [dataset_label(ds) for ds in counts.index], rotation=20, ha="right")
    ax.set_ylabel("Trials")
    panel_label(ax, "a")
    ax.legend(frameon=False)

    ax = axes[1]
    phase_counts = pd.crosstab(phase_trials["dataset"], phase_trials["skill_category"]).reindex(DATASET_ORDER).fillna(0)
    bottom = np.zeros(len(phase_counts))
    x = np.arange(len(phase_counts))
    for skill in SKILL_ORDER:
        vals = phase_counts.get(skill, pd.Series(0, index=phase_counts.index)).to_numpy()
        ax.bar(x, vals, bottom=bottom, color=SKILL_COLORS[skill], width=0.7)
        bottom += vals
    ax.set_xticks(x, [dataset_label(ds) for ds in phase_counts.index], rotation=20, ha="right")
    ax.set_ylabel("Trials")
    panel_label(ax, "b")
    fig.tight_layout()
    save_fig(fig, "fig1_cohort_structure")


def plot_task_setup_overview() -> None:
    sample_paths = [
        FIG / "source" / "task_frames" / "bapes_trial17_frame00300.png",
        FIG / "source" / "task_frames" / "urology1_test13_frame00100.png",
        FIG / "source" / "task_frames" / "urology2_trial34_frame00500.png",
    ]
    if not all(path.exists() for path in sample_paths):
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    for ax, path, panel in zip(axes, sample_paths, "abc"):
        show_center_crop(ax, path)
        image_panel_label(ax, panel, y=-0.07)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.12, wspace=0.03)
    save_fig(fig, "fig0_task_setup_overview")


def plot_training_systems_context() -> None:
    task_paths = [
        FIG / "source" / "training_tasks" / "fls_peg_transfer.png",
        FIG / "source" / "training_tasks" / "fls_circle_cutting.png",
        FIG / "source" / "training_tasks" / "fls_ligation_loop.png",
        FIG / "source" / "training_tasks" / "fls_extracorporeal_knot.png",
        FIG / "source" / "training_tasks" / "fls_intracorporeal_knot.png",
        FIG / "source" / "training_tasks" / "eblus_needle_guidance.png",
    ]
    if not all(path.exists() for path in task_paths):
        return
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.25))
    for ax, path, panel in zip(axes.flat, task_paths, "abcdef"):
        show_center_crop(ax, path)
        image_panel_label(ax, panel, y=-0.07)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.07, wspace=0.03, hspace=0.18)
    save_fig(fig, "fig0_training_systems")


def plot_collection_setups() -> None:
    setup_paths = [
        FIG / "source" / "collection_setups" / "setup_bapes_2024.jpg",
        FIG / "source" / "collection_setups" / "setup_urology_2023.jpeg",
        FIG / "source" / "collection_setups" / "setup_urology_2024.jpeg",
        FIG / "source" / "collection_setups" / "setup_urology_2025.jpeg",
    ]
    if not all(path.exists() for path in setup_paths):
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.45))
    for ax, path, panel in zip(axes.flat, setup_paths, "abcd"):
        show_center_crop(ax, path)
        image_panel_label(ax, panel, y=-0.06)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.07, wspace=0.025, hspace=0.15)
    save_fig(fig, "fig0_collection_setups")


def plot_peg_transfer_cycle_placeholder() -> None:
    stages = [
        ("Reach", "Approach the source peg", "left", (0.24, 0.54), (0.22, 0.45)),
        ("Grasp", "Close jaws around object", "left", (0.28, 0.55), (0.28, 0.55)),
        ("Lift", "Lift clear of the peg", "left", (0.30, 0.68), (0.30, 0.68)),
        ("Transfer", "Meet the opposite tool", "both", (0.50, 0.65), (0.50, 0.65)),
        ("Place", "Move to the target peg", "right", (0.72, 0.58), (0.72, 0.58)),
        ("Release", "Release and reset", "right", (0.76, 0.52), (0.78, 0.45)),
    ]

    def draw_scene(ax: plt.Axes, stage: tuple[str, str, str, tuple[float, float], tuple[float, float]], panel: str) -> None:
        name, subtitle, active, tool_tip, object_xy = stage
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.add_patch(Rectangle((0.10, 0.20), 0.80, 0.50, facecolor="#F8F8F8", edgecolor="#777777", lw=0.8))
        for x in (0.25, 0.75):
            ax.plot([x, x], [0.34, 0.56], color="#555555", lw=2.0, solid_capstyle="round")
            ax.add_patch(plt.Circle((x, 0.56), 0.025, facecolor="#D0D0D0", edgecolor="#777777", lw=0.6))

        left_tip = tool_tip if active in {"left", "both"} else (0.18, 0.40)
        right_tip = tool_tip if active in {"right", "both"} else (0.82, 0.40)
        ax.plot([0.05, left_tip[0]], [0.22, left_tip[1]], color="#0072B2", lw=2.2, solid_capstyle="round")
        ax.plot([0.95, right_tip[0]], [0.22, right_tip[1]], color="#009E73", lw=2.2, solid_capstyle="round")

        if active == "both":
            ax.add_patch(FancyArrowPatch((0.35, 0.63), (0.65, 0.63), arrowstyle="<->", mutation_scale=10, lw=1.2, color="#555555"))
        elif active == "left":
            ax.add_patch(FancyArrowPatch((0.18, 0.45), tool_tip, arrowstyle="->", mutation_scale=10, lw=1.2, color="#0072B2"))
        elif active == "right":
            ax.add_patch(FancyArrowPatch((0.58, 0.60), tool_tip, arrowstyle="->", mutation_scale=10, lw=1.2, color="#009E73"))

        ax.add_patch(plt.Circle(object_xy, 0.040, facecolor="#E69F00", edgecolor="#8A5A00", lw=0.8))
        if name == "Release":
            ax.add_patch(FancyArrowPatch((0.70, 0.52), (0.80, 0.52), arrowstyle="->", mutation_scale=9, lw=1.0, color="#CC3311"))
            ax.text(0.50, 0.16, "Optional nudge or correction", ha="center", va="center", fontsize=6.4, color="#CC3311")

        ax.text(0.50, 0.91, name, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(0.50, 0.82, subtitle, ha="center", va="center", fontsize=6.9, color="#333333", wrap=True)
        image_panel_label(ax, panel, y=-0.02)

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8))
    for ax, stage, panel in zip(axes.flat, stages, "abcdef"):
        draw_scene(ax, stage, panel)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.06, wspace=0.04, hspace=0.22)
    save_fig(fig, "fig0_peg_transfer_cycle_placeholder")


def plot_feature_effects(stats_df: pd.DataFrame) -> None:
    d = stats_df.dropna(subset=["cliffs_delta_novice_expert"]).copy()
    d = d[d["n_nonmissing"] >= 20]
    orientation = []
    for feature in d["feature"]:
        arrow, _ = FEATURE_DIRECTIONS.get(feature, ("$\\uparrow$", ""))
        orientation.append(-1 if "\\uparrow" in arrow else 1)
    d["_orientation"] = orientation
    d["oriented_delta"] = d["cliffs_delta_novice_expert"] * d["_orientation"]
    d["oriented_ci_low"] = np.where(
        d["_orientation"] > 0,
        d["delta_ci_low"],
        -d["delta_ci_high"],
    )
    d["oriented_ci_high"] = np.where(
        d["_orientation"] > 0,
        d["delta_ci_high"],
        -d["delta_ci_low"],
    )
    d = d.loc[d["oriented_delta"].abs().sort_values(ascending=False).index].head(18)
    d = d.sort_values("oriented_delta")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.2, max(5.2, 0.22 * len(d))))
    ax.axvline(0, color="#444444", lw=0.8)
    colors = ["#0072B2" if q < 0.05 else "#8A8A8A" for q in d["kruskal_q"]]
    ax.errorbar(
        d["oriented_delta"],
        y,
        xerr=[
            d["oriented_delta"] - d["oriented_ci_low"],
            d["oriented_ci_high"] - d["oriented_delta"],
        ],
        fmt="none",
        ecolor="#666666",
        elinewidth=0.9,
        capsize=2,
    )
    ax.scatter(d["oriented_delta"], y, s=24, c=colors, zorder=3)
    ax.set_yticks(y, d["label"])
    ax.set_xlabel("Oriented Cliff's delta for expert versus novice")
    ax.text(
        0.01,
        1.01,
        "Novice-favouring",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
    )
    ax.text(
        0.99,
        1.01,
        "Expert-favouring",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout()
    save_fig(fig, "fig2_feature_effects")


def plot_phase_timing(cycles: pd.DataFrame, trials: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    recording_counts = (
        trials[["trial_key", "trial_id"]]
        .drop_duplicates()
        .groupby("trial_key")["trial_id"]
        .transform("size")
    )
    unique_keys = set(
        trials[["trial_key", "trial_id"]]
        .drop_duplicates()
        .loc[recording_counts.eq(1), "trial_key"]
    )
    inferential_trials = trials[trials["trial_key"].isin(unique_keys)].copy()
    inferential_cycles = cycles[cycles["trial_key"].isin(unique_keys)].copy()
    trial_cycles = (
        inferential_cycles.groupby(["dataset", "skill_category", "trial_id"], as_index=False)
        .agg(mean_cycle_duration_s=("duration_s", "mean"))
    )
    ax = axes[0]
    x_base = np.arange(len(DATASET_ORDER))
    offsets = {"novice": -0.22, "intermediate": 0.0, "expert": 0.22}
    for skill in SKILL_ORDER:
        xs, ys, los, his = [], [], [], []
        for i, ds in enumerate(DATASET_ORDER):
            g = trial_cycles[
                (trial_cycles["dataset"] == ds)
                & (trial_cycles["skill_category"] == skill)
            ]["mean_cycle_duration_s"].dropna()
            if len(g) == 0:
                continue
            lo, hi = bootstrap_ci(g)
            xs.append(i + offsets[skill])
            ys.append(g.mean())
            los.append(g.mean() - lo)
            his.append(hi - g.mean())
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o", capsize=2, color=SKILL_COLORS[skill], label=nice(skill), ms=4)
    ax.set_xticks(x_base, [dataset_label(ds) for ds in DATASET_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Cycle duration (s)")
    panel_label(ax, "a")
    ax.legend(frameon=False)

    ax = axes[1]
    phase_cols = ["frac_reach", "frac_grasp", "frac_transfer", "frac_place", "frac_nudge", "frac_idle"]
    means = inferential_trials.groupby("skill_category")[phase_cols].mean().reindex(SKILL_ORDER)
    bottom = np.zeros(len(means))
    x = np.arange(len(means))
    for c in phase_cols:
        phase = c.replace("frac_", "")
        vals = means[c].fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottom, color=PHASE_COLORS[phase], label=nice(phase), width=0.7)
        bottom += vals
    ax.set_xticks(x, [nice(s) for s in SKILL_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Mean fraction of corrected trial time")
    panel_label(ax, "b")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.25, wspace=0.34)
    save_fig(fig, "fig3_phase_timing")


def plot_experience_links(df: pd.DataFrame, phase_trials: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.2), sharex=False)
    for c, ds in enumerate(DATASET_ORDER):
        ax = axes[0, c]
        g = df[df["dataset"] == ds]
        ax.scatter(
            np.log10(g["total_procedures"] + 1),
            g["bimanual_correlation"],
            s=18,
            alpha=0.75,
            color=DATASET_COLORS[ds],
        )
        ax.set_title(dataset_label(ds), fontsize=8)
        ax.set_xlabel("log10(total procedures + 1)")
        if c == 0:
            ax.set_ylabel("Bimanual correlation")
        panel_label(ax, chr(ord("a") + c))

        ax = axes[1, c]
        g = phase_trials[phase_trials["dataset"] == ds]
        ax.scatter(
            np.log10(g["total_procedures"] + 1),
            g["mean_cycle_s"],
            s=22,
            alpha=0.8,
            color=DATASET_COLORS[ds],
        )
        ax.set_xlabel("log10(total procedures + 1)")
        if c == 0:
            ax.set_ylabel("Mean cycle duration (s)")
        panel_label(ax, chr(ord("d") + c))
    fig.tight_layout()
    save_fig(fig, "fig4_experience_links")


def plot_osats_proxy(proxy: pd.DataFrame, clusters: pd.DataFrame, cluster_summary: dict[str, object]) -> None:
    band_proxy = proxy.merge(
        clusters[["dataset", "trial_name", "metric_cluster"]].drop_duplicates(
            ["dataset", "trial_name"]
        ),
        on=["dataset", "trial_name"],
        how="left",
        validate="one_to_one",
    )
    domain_cols = [
        "Coordination-control composite",
        *CORE_PERFORMANCE_DOMAINS,
        "Task efficiency",
        "Workspace excursion",
    ]
    display_labels = {
        "Bimanual coordination": "Bimanual\ncoordination",
        "Task efficiency": "Task\nefficiency",
        "Instrument motion control": "Instrument\nmotion control",
        "Workspace excursion": "Workspace\ncompactness\n(descriptive)",
        "Coordination-control composite": "Coordination and control\nscore",
        "Analysed duration (s)": "Analysed\nduration",
    }

    fig, axes = plt.subplots(len(DATASET_ORDER), 1, figsize=(7.2, 6.8), sharex=True)
    ims = []
    for idx, (ax, ds, label) in enumerate(zip(axes, DATASET_ORDER, "abc")):
        cohort = band_proxy[band_proxy["dataset"] == ds]
        summary = cohort.groupby("metric_cluster")[domain_cols].mean().reindex(BAND_ORDER)
        im = ax.imshow(summary[domain_cols].to_numpy(float), vmin=1, vmax=5, cmap="YlGnBu", aspect="auto")
        ims.append(im)
        counts = cohort["metric_cluster"].value_counts()
        ax.set_title(dataset_label(ds), fontsize=8, loc="left", pad=3)
        ax.set_yticks(
            np.arange(len(summary)),
            [f"{BAND_SHORT_LABELS[b]}  (n={int(counts.get(b, 0))})" for b in summary.index],
        )
        ax.set_xticks(np.arange(len(domain_cols)))
        if idx == len(DATASET_ORDER) - 1:
            ax.set_xticklabels([display_labels[c] for c in domain_cols], rotation=24, ha="right")
        else:
            ax.set_xticklabels([])
        ax.set_ylabel("Measured\nband")
        for r in range(summary.shape[0]):
            for c in range(summary.shape[1]):
                value = summary.iloc[r, c]
                text = "Not recorded" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(c, r, text, ha="center", va="center", fontsize=6.1)
        panel_label(ax, label)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.24, hspace=0.38)
    cax = fig.add_axes([0.30, 0.055, 0.42, 0.025])
    cbar = fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    cbar.set_label("Relative domain score (1-5)")
    save_fig(fig, "fig5_osats_proxy_heatmap")

    held_out_cols = [
        "Task efficiency",
        "Analysed duration (s)",
        "Workspace excursion",
    ]
    identifier_counts = band_proxy.groupby("trial_key")["trial_key"].transform("size")
    held_out_proxy = band_proxy[identifier_counts == 1].copy()
    fig, axes = plt.subplots(
        len(held_out_cols),
        len(DATASET_ORDER),
        figsize=(9.4, 6.8),
        sharey="row",
    )
    for r, domain in enumerate(held_out_cols):
        for c, ds in enumerate(DATASET_ORDER):
            ax = axes[r, c]
            cohort = held_out_proxy[held_out_proxy["dataset"] == ds]
            data = [cohort.loc[cohort["metric_cluster"] == band, domain].dropna() for band in BAND_ORDER]
            if sum(len(values) for values in data) == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "Jaw aperture not recorded", ha="center", va="center", transform=ax.transAxes, fontsize=7)
                continue
            bp = ax.boxplot(
                data,
                patch_artist=True,
                tick_labels=[BAND_SHORT_LABELS[band] for band in BAND_ORDER],
                showfliers=False,
                widths=0.58,
            )
            for patch, band in zip(bp["boxes"], BAND_ORDER):
                patch.set_facecolor(CLUSTER_COLORS[band])
                patch.set_alpha(0.68)
            for i, values in enumerate(data):
                if len(values) == 0:
                    continue
                xs = np.full(len(values), i + 1) + np.linspace(-0.08, 0.08, len(values))
                ax.scatter(xs, values, s=6.5, color="#222222", alpha=0.34, linewidths=0)
            if domain != "Analysed duration (s)":
                ax.set_ylim(1, 5)
            if r == 0:
                ax.set_title(dataset_label(ds), fontsize=8)
            if c == 0:
                unit = "score (1-5)" if domain != "Analysed duration (s)" else "seconds"
                ax.set_ylabel(f"{display_labels[domain].replace(chr(10), ' ')}\n{unit}")
            ax.tick_params(axis="x", rotation=0, labelsize=6.5)
            ax.text(
                0.08,
                1.03,
                chr(ord("a") + r * len(DATASET_ORDER) + c),
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
            )
    fig.tight_layout(h_pad=1.12, w_pad=0.52)
    save_fig(fig, "fig5_osats_proxy_boxplots")

    experience_cols = [
        *CORE_PERFORMANCE_DOMAINS,
        "Task efficiency",
        "Workspace excursion",
        "Coordination-control composite",
    ]
    fig, axes = plt.subplots(len(DATASET_ORDER), 1, figsize=(7.0, 6.8), sharex=True)
    ims = []
    for idx, (ax, ds, label) in enumerate(zip(axes, DATASET_ORDER, "abc")):
        cohort = proxy[proxy["dataset"] == ds]
        summary = cohort.groupby("skill_category")[experience_cols].mean().reindex(SKILL_ORDER)
        im = ax.imshow(summary[experience_cols].to_numpy(float), vmin=1, vmax=5, cmap="YlGnBu", aspect="auto")
        ims.append(im)
        ax.set_title(dataset_label(ds), fontsize=8, loc="left", pad=3)
        ax.set_yticks(np.arange(len(summary)), [nice(s) for s in summary.index])
        ax.set_xticks(np.arange(len(experience_cols)))
        if idx == len(DATASET_ORDER) - 1:
            ax.set_xticklabels([display_labels.get(c, c) for c in experience_cols], rotation=24, ha="right")
        else:
            ax.set_xticklabels([])
        ax.set_ylabel("Procedure-count\ngroup")
        for rr in range(summary.shape[0]):
            for cc in range(summary.shape[1]):
                value = summary.iloc[rr, cc]
                ax.text(cc, rr, "Not recorded" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center", fontsize=6.1)
        panel_label(ax, label)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.24, hspace=0.38)
    cax = fig.add_axes([0.30, 0.055, 0.42, 0.025])
    cbar = fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    cbar.set_label("Relative domain score (1-5)")
    save_fig(fig, "figS_experience_domain_heatmap")

    fig, axes = plt.subplots(len(experience_cols), len(DATASET_ORDER), figsize=(9.4, 11.2), sharey=True)
    for r, domain in enumerate(experience_cols):
        for c, ds in enumerate(DATASET_ORDER):
            ax = axes[r, c]
            cohort = proxy[proxy["dataset"] == ds]
            data = [cohort.loc[cohort["skill_category"] == skill, domain].dropna() for skill in SKILL_ORDER]
            if sum(len(values) for values in data) == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "Jaw aperture not recorded", ha="center", va="center", transform=ax.transAxes, fontsize=7)
                continue
            bp = ax.boxplot(data, patch_artist=True, tick_labels=[nice(s) for s in SKILL_ORDER], showfliers=False, widths=0.58)
            for patch, skill in zip(bp["boxes"], SKILL_ORDER):
                patch.set_facecolor(SKILL_COLORS[skill])
                patch.set_alpha(0.68)
            for i, values in enumerate(data):
                if len(values) == 0:
                    continue
                xs = np.full(len(values), i + 1) + np.linspace(-0.08, 0.08, len(values))
                ax.scatter(xs, values, s=6.5, color="#222222", alpha=0.34, linewidths=0)
            ax.set_ylim(1, 5)
            if r == 0:
                ax.set_title(dataset_label(ds), fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{display_labels.get(domain, domain).replace(chr(10), ' ')}\nscore (1-5)")
            ax.tick_params(axis="x", rotation=20, labelsize=6.2)
            panel_label(ax, chr(ord("a") + r * len(DATASET_ORDER) + c))
    fig.tight_layout(h_pad=1.08, w_pad=0.50)
    save_fig(fig, "figS_experience_domain_boxplots")


def plot_kmeans_clusters(clusters: pd.DataFrame, summary: dict[str, object]) -> None:
    if clusters.empty or not summary.get("available"):
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.35), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    ax = axes[0]
    for i, cl in enumerate(BAND_ORDER):
        g = clusters[clusters["metric_cluster"] == cl].sort_values(
            "Coordination-control composite"
        )
        if g.empty:
            continue
        y = np.linspace(i - 0.16, i + 0.16, len(g))
        ax.scatter(
            g["Coordination-control composite"],
            y,
            s=21,
            alpha=0.82,
            color=CLUSTER_COLORS[cl],
            label=BAND_SHORT_LABELS[cl],
        )
    for threshold in summary.get("metric_band_thresholds", []):
        ax.axvline(threshold, color="#555555", lw=0.9, ls="--")
    ax.set_xlim(1, 5)
    ax.set_yticks(range(len(BAND_ORDER)), [BAND_SHORT_LABELS[b] for b in BAND_ORDER])
    ax.set_xlabel("Coordination and control score (1-5)")
    ax.set_ylabel("Relative band")
    panel_label(ax, "a")

    ax = axes[1]
    validation = pd.DataFrame(summary.get("k_validation", []))
    if not validation.empty:
        ax.plot(validation["k"], validation["silhouette"], color="#555555", marker="o", lw=1.1, ms=4)
        selected = validation[validation["k"] == 3]
        if not selected.empty:
            ax.scatter(selected["k"], selected["silhouette"], s=42, color="#CC3311", zorder=3)
    ax.set_xticks(range(2, 7))
    ax.set_xlabel("Number of bands, k")
    ax.set_ylabel("Silhouette score")
    panel_label(ax, "b")

    ax = axes[2]
    for i, cl in enumerate(BAND_ORDER):
        g = clusters[clusters["metric_cluster"] == cl]
        if g.empty:
            continue
        values = np.log10(g["total_procedures"] + 1)
        xs = np.full(len(g), i) + np.linspace(-0.10, 0.10, len(g))
        ax.scatter(xs, values, s=18, alpha=0.78, color=CLUSTER_COLORS[cl])
        ax.hlines(values.median(), i - 0.24, i + 0.24, colors="#222222", lw=1.2)
    ax.set_xticks(range(len(BAND_ORDER)), [BAND_SHORT_LABELS[cl] for cl in BAND_ORDER])
    ax.set_ylabel("log10(total procedures + 1)")
    ax.set_xlabel("Relative band")
    panel_label(ax, "c")
    fig.tight_layout()
    save_fig(fig, "fig6_kmeans_procedure_thresholds")


def plot_motion_domains(df: pd.DataFrame) -> None:
    panels = [
        ("Path length", "tool1_path_length", "tool2_path_length", "mm"),
        ("Average speed", "tool1_avg_speed", "tool2_avg_speed", "mm/s"),
        ("Speed variability", "tool1_speed_cv", "tool2_speed_cv", "CV"),
        ("Acceleration variability", "tool1_accel_cv", "tool2_accel_cv", "CV"),
        ("Normalised jerk", "tool1_normalized_jerk", "tool2_normalized_jerk", "a.u."),
        ("Rotation per path", "tool1_rotation_per_path", "tool2_rotation_per_path", "degrees/mm"),
        ("Angular velocity variability", "tool1_angular_velocity_cv", "tool2_angular_velocity_cv", "CV"),
        ("Workspace depth range", "tool1_range_z", "tool2_range_z", "mm"),
        ("Working area", "tool1_working_area_xy", "tool2_working_area_xy", "mm$^2$"),
        ("Movement count", "tool1_num_movements", "tool2_num_movements", "count"),
        ("Idle episodes", "tool1_idle_episode_count", "tool2_idle_episode_count", "count"),
        ("Speed peaks", "tool1_num_speed_peaks", "tool2_num_speed_peaks", "count"),
    ]
    for ds in DATASET_ORDER:
        cohort = df[df["dataset"] == ds]
        if cohort.empty:
            continue
        fig, axes = plt.subplots(3, 4, figsize=(9.4, 6.7))
        axes = axes.ravel()
        for ax, (title, c1, c2, unit), label in zip(axes, panels, list("abcdefghijkl")):
            plot_df = cohort[["skill_category", c1, c2]].copy()
            plot_df["metric"] = plot_df[[c1, c2]].mean(axis=1, skipna=True)
            data = [plot_df.loc[plot_df["skill_category"] == s, "metric"].dropna() for s in SKILL_ORDER]
            bp = ax.boxplot(data, patch_artist=True, tick_labels=[nice(s) for s in SKILL_ORDER], showfliers=False)
            for patch, skill in zip(bp["boxes"], SKILL_ORDER):
                patch.set_facecolor(SKILL_COLORS[skill])
                patch.set_alpha(0.65)
            ax.set_ylabel(unit)
            ax.set_title(title, fontsize=8)
            ax.tick_params(axis="x", rotation=20)
            panel_label(ax, label)
        for ax in axes[len(panels) :]:
            ax.axis("off")
        fig.tight_layout()
        save_fig(fig, f"fig7_motion_component_domains_{dataset_slug(ds)}")


def plot_jaw_and_rotation(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    jaw_df = df[df["dataset"].isin(["7DOF2024", "BAPES2024"])].copy()
    panels = [
        ("Jaw signal range", "tool1_jaw_angle_range", "tool2_jaw_angle_range", "Voltage range"),
        ("Detected aperture cycles", "tool1_jaw_open_close_count", "tool2_jaw_open_close_count", "Cycles"),
        ("Angular velocity variability", "tool1_angular_velocity_cv", "tool2_angular_velocity_cv", "CV"),
    ]
    for ax, (title, c1, c2, ylabel), label in zip(axes, panels, ["a", "b", "c"]):
        source = jaw_df if "jaw" in c1 else df
        order = ["BAPES2024", "7DOF2024"] if "jaw" in c1 else DATASET_ORDER
        source = source.copy()
        source["metric"] = source[[c1, c2]].mean(axis=1, skipna=True)
        for i, ds in enumerate(order):
            g = source[source["dataset"] == ds]["metric"].dropna()
            if len(g) == 0:
                continue
            x = np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g))
            ax.scatter(x, g, s=15, alpha=0.75, color=DATASET_COLORS[ds])
            lo, hi = bootstrap_ci(g)
            ax.errorbar(i, g.mean(), yerr=[[g.mean() - lo], [hi - g.mean()]], color="#333333", capsize=3, fmt="o", ms=3)
        ax.set_xticks(range(len(order)), [dataset_label(ds) for ds in order], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        panel_label(ax, label)
    fig.tight_layout()
    save_fig(fig, "fig8_jaw_rotation_components")


def plot_workspace_heatmaps(motion: pd.DataFrame) -> None:
    if motion.empty:
        return
    long_rows = []
    for tool in ["tool1", "tool2"]:
        sub = motion[["dataset", "cohort_label", "skill_category", f"{tool}_x", f"{tool}_y", f"{tool}_z"]].copy()
        sub.columns = ["dataset", "cohort_label", "skill_category", "x", "y", "z"]
        sub["tool"] = tool
        long_rows.append(sub)
    pos = pd.concat(long_rows, ignore_index=True).dropna(subset=["x", "y", "z"])
    planes = [("x", "y", "Camera X (mm)", "Camera Y (mm)"), ("x", "z", "Camera X (mm)", "Depth Z (mm)"), ("y", "z", "Camera Y (mm)", "Depth Z (mm)")]
    fig, axes = plt.subplots(len(planes), len(DATASET_ORDER), figsize=(7.4, 6.4), sharex=False, sharey=False)
    mesh = None
    for r, (a, b, xlabel, ylabel) in enumerate(planes):
        for c, ds in enumerate(DATASET_ORDER):
            ax = axes[r, c]
            g = pos[pos["dataset"] == ds]
            density, a_edges, b_edges = np.histogram2d(g[a], g[b], bins=48)
            peak = density.max()
            relative_density = density / peak if peak > 0 else density
            relative_density = np.ma.masked_where(relative_density <= 0, relative_density)
            mesh = ax.pcolormesh(
                a_edges,
                b_edges,
                relative_density.T,
                cmap="viridis",
                vmin=0,
                vmax=1,
                shading="auto",
            )
            if r == 0:
                ax.set_title(dataset_label(ds), fontsize=8)
            if c == 0:
                ax.set_ylabel(ylabel)
                panel_label(ax, chr(ord("a") + r))
            ax.set_xlabel(xlabel if r == len(planes) - 1 else "")
            ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.16, wspace=0.24, hspace=0.28)
    if mesh is not None:
        cax = fig.add_axes([0.30, 0.055, 0.40, 0.018])
        colourbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        colourbar.set_label("Relative sampling density within each panel", fontsize=7)
        colourbar.ax.tick_params(labelsize=6)
    save_fig(fig, "fig9_workspace_heatmaps_camera_planes")


def plot_skill_workspace_xy(motion: pd.DataFrame) -> None:
    if motion.empty:
        return
    rows = []
    for tool in ["tool1", "tool2"]:
        sub = motion[["dataset", "skill_category", f"{tool}_x", f"{tool}_y"]].copy()
        sub.columns = ["dataset", "skill_category", "x", "y"]
        rows.append(sub)
    pos = pd.concat(rows, ignore_index=True).dropna()
    fig, axes = plt.subplots(len(SKILL_ORDER), len(DATASET_ORDER), figsize=(7.4, 6.4), sharex=False, sharey=False)
    mesh = None
    for r, skill in enumerate(SKILL_ORDER):
        for c, ds in enumerate(DATASET_ORDER):
            ax = axes[r, c]
            g = pos[(pos["dataset"] == ds) & (pos["skill_category"] == skill)]
            if len(g):
                density, x_edges, y_edges = np.histogram2d(g["x"], g["y"], bins=42)
                peak = density.max()
                relative_density = density / peak if peak > 0 else density
                relative_density = np.ma.masked_where(relative_density <= 0, relative_density)
                mesh = ax.pcolormesh(
                    x_edges,
                    y_edges,
                    relative_density.T,
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                    shading="auto",
                )
            if r == 0:
                ax.set_title(dataset_label(ds), fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{nice(skill)}\nCamera Y (mm)")
            if r == len(SKILL_ORDER) - 1:
                ax.set_xlabel("Camera X (mm)")
            ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.16, wspace=0.24, hspace=0.28)
    if mesh is not None:
        cax = fig.add_axes([0.30, 0.055, 0.40, 0.018])
        colourbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        colourbar.set_label("Relative sampling density within each panel", fontsize=7)
        colourbar.ax.tick_params(labelsize=6)
    save_fig(fig, "fig10_skill_workspace_xy_heatmaps")


def plot_phase_by_cluster(trials: pd.DataFrame, clusters: pd.DataFrame) -> None:
    if clusters.empty or "metric_cluster" not in clusters:
        return
    phase_cols = ["frac_reach", "frac_grasp", "frac_transfer", "frac_place", "frac_nudge", "frac_idle"]
    phase = trials.copy()
    phase["trial_key"] = [phase_trial_key(ds, ts) for ds, ts in zip(phase["dataset"], phase["trial_short"])]
    cluster_lookup = clusters[["trial_key", "metric_cluster"]].drop_duplicates("trial_key")
    merged = phase.merge(cluster_lookup, on="trial_key", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.65))
    for ax, group_col, labels, label in [
        (axes[0], "skill_category", SKILL_ORDER, "a"),
        (axes[1], "metric_cluster", BAND_ORDER, "b"),
    ]:
        means = merged.groupby(group_col)[phase_cols].mean().reindex(labels)
        left = np.zeros(len(means))
        y = np.arange(len(means))
        for col in phase_cols:
            phase = col.replace("frac_", "")
            vals = means[col].fillna(0).to_numpy()
            ax.barh(y, vals, left=left, color=PHASE_COLORS[phase], label=nice(phase), height=0.68)
            left += vals
        tick_labels = [BAND_SHORT_LABELS.get(s, nice(s)) for s in labels]
        ax.set_yticks(y, tick_labels, fontsize=7)
        ax.set_xlabel("Mean fraction of trial time", fontsize=8)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        panel_label(ax, label)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, 0.03))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.90, bottom=0.30, wspace=0.36)
    save_fig(fig, "fig11_phase_allocation_original_vs_cluster")


def plot_ssl_embeddings(ssl_df: pd.DataFrame) -> None:
    if ssl_df.empty or "ssl_pc1" not in ssl_df.columns:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.25))
    ax = axes[0]
    for ds in DATASET_ORDER:
        g = ssl_df[ssl_df["dataset"] == ds]
        ax.scatter(g["ssl_pc1"], g["ssl_pc2"], s=18, alpha=0.8, color=DATASET_COLORS[ds], label=dataset_label(ds))
    ax.set_xlabel("Embedding PC1")
    ax.set_ylabel("Embedding PC2")
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "a")

    ax = axes[1]
    for skill in SKILL_ORDER:
        g = ssl_df[ssl_df["skill_category"] == skill]
        ax.scatter(g["ssl_pc1"], g["ssl_pc2"], s=18, alpha=0.75, color=SKILL_COLORS[skill], label=nice(skill))
    ax.set_xlabel("Embedding PC1")
    ax.set_ylabel("Embedding PC2")
    panel_label(ax, "b")

    ax = axes[2]
    sc = ax.scatter(
        ssl_df["ssl_pc1"],
        ssl_df["ssl_pc2"],
        c=np.log10(ssl_df["total_procedures"] + 1),
        cmap="viridis",
        s=18,
        alpha=0.8,
    )
    ax.set_xlabel("Embedding PC1")
    ax.set_ylabel("Embedding PC2")
    panel_label(ax, "c")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.34, wspace=0.34)
    cax = fig.add_axes([0.32, 0.11, 0.36, 0.045])
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("log10(total procedures + 1)")
    save_fig(fig, "fig12_unsupervised_motion_embeddings")


def plot_tool_specific_motion_boxplots(df: pd.DataFrame) -> None:
    panels = [
        ("Path length", "path_length", "mm", False),
        ("Average speed", "avg_speed", "mm/s", True),
        ("Speed variability", "speed_cv", "CV", False),
        ("Normalised jerk", "normalized_jerk", "a.u.", False),
        ("Rotation per path", "rotation_per_path", "degrees/mm", False),
        ("Depth range", "range_z", "mm", False),
        ("Working area", "working_area_xy", "mm$^2$", False),
        ("Movement count", "num_movements", "count", False),
    ]
    offsets = {"novice": -0.22, "intermediate": 0.0, "expert": 0.22}
    for ds in DATASET_ORDER:
        cohort = df[df["dataset"] == ds]
        if cohort.empty:
            continue
        fig, axes = plt.subplots(3, 3, figsize=(9.1, 7.4), sharex=False)
        axes = axes.ravel()
        for ax, (title, suffix, ylabel, higher_is_more), label in zip(axes, panels, list("abcdefgh")):
            positions = []
            data = []
            colors = []
            for base, tool in enumerate(["tool1", "tool2"]):
                col = f"{tool}_{suffix}"
                if col not in cohort.columns:
                    continue
                for skill in SKILL_ORDER:
                    values = cohort.loc[cohort["skill_category"] == skill, col].dropna()
                    data.append(values)
                    positions.append(base + offsets[skill])
                    colors.append(SKILL_COLORS[skill])
            if not data:
                ax.axis("off")
                continue
            bp = ax.boxplot(data, positions=positions, widths=0.16, patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
            direction = "$\\uparrow$" if higher_is_more else "$\\downarrow$"
            ax.set_title(f"{title} ({direction})", fontsize=8)
            ax.set_xticks([0, 1], ["Left tool", "Right tool"])
            ax.set_ylabel(ylabel)
            panel_label(ax, label)
        for ax in axes[len(panels) :]:
            ax.axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color=SKILL_COLORS[s], alpha=0.65) for s in SKILL_ORDER]
        fig.legend(handles, [nice(s) for s in SKILL_ORDER], frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.02))
        fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.12, wspace=0.34, hspace=0.44)
        save_fig(fig, f"fig13_tool_specific_motion_boxplots_{dataset_slug(ds)}")


def plot_feature_importance() -> None:
    path = OUT / "skill_feature_importance.csv"
    if not path.exists():
        return
    imp = pd.read_csv(path).head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    y = np.arange(len(imp))
    ax.barh(y, imp["importance"], color="#4C78A8")
    ax.set_yticks(y, imp["label"])
    ax.set_xlabel("Random forest feature importance")
    ax.set_ylabel("Kinematic feature")
    fig.tight_layout()
    save_fig(fig, "fig14_skill_feature_importance")


def plot_sorted_participant_bars(df: pd.DataFrame, proxy: pd.DataFrame) -> None:
    merged = df.copy()
    merged["Coordination-control composite"] = (
        proxy["Coordination-control composite"].to_numpy() if len(proxy) == len(df) else np.nan
    )
    specs = [
        ("Coordination-control composite", "Coordination and\ncontrol score ($\\uparrow$)", True),
        ("bimanual_correlation", "Bimanual\ncorrelation ($\\uparrow$)", True),
        ("combined_idle_ratio", "Idle fraction ($\\downarrow$)", False),
        ("tool1_normalized_jerk", "Left tool normalised\njerk ($\\downarrow$)", False),
        ("tool2_normalized_jerk", "Right tool normalised\njerk ($\\downarrow$)", False),
        ("total_time", "Analysed\nduration ($\\downarrow$)", False),
    ]
    fig, axes = plt.subplots(len(specs), len(DATASET_ORDER), figsize=(9.4, 9.2), sharex=False)
    for r, (col, ylabel, higher_better) in enumerate(specs):
        for c, ds in enumerate(DATASET_ORDER):
            ax = axes[r, c]
            if col not in merged.columns:
                ax.axis("off")
                continue
            d = (
                merged.loc[merged["dataset"] == ds, ["skill_category", "dataset", col]]
                .dropna()
                .sort_values(col, ascending=not higher_better)
            )
            colors = [SKILL_COLORS.get(s, "#999999") for s in d["skill_category"]]
            ax.bar(np.arange(len(d)), d[col], color=colors, width=0.85)
            if c == 0:
                ax.set_ylabel(
                    ylabel,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=24,
                    fontsize=7,
                )
            if r == 0:
                ax.set_title(dataset_label(ds), fontsize=8)
            ax.set_xticks([])
            panel_label(ax, chr(ord("a") + r * len(DATASET_ORDER) + c))
    handles = [plt.Rectangle((0, 0), 1, 1, color=SKILL_COLORS[s]) for s in SKILL_ORDER]
    fig.legend(handles, [nice(s) for s in SKILL_ORDER], frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.20, right=0.99, top=0.94, bottom=0.08, wspace=0.24, hspace=0.46)
    save_fig(fig, "fig15_sorted_trial_metric_bars")


def _normalised_cohort_means(df: pd.DataFrame, specs: list[tuple[str, str, int]]) -> pd.DataFrame:
    rows = []
    for col, label, direction in specs:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce") * direction
        finite = x[np.isfinite(x)]
        if finite.empty:
            continue
        lo, hi = finite.quantile([0.05, 0.95])
        denom = hi - lo if hi != lo else 1.0
        scaled = ((x - lo) / denom).clip(0, 1)
        for ds in DATASET_ORDER:
            rows.append({"dataset": ds, "metric": label, "value": float(scaled[df["dataset"] == ds].mean())})
    return pd.DataFrame(rows)


def plot_cohort_radar(df: pd.DataFrame) -> None:
    specs = [
        ("bimanual_correlation", "Bimanual\nsynchrony", 1),
        ("simultaneous_motion_ratio", "Simultaneous\nmotion", 1),
        ("combined_idle_ratio", "Low idle\ntime", -1),
        ("total_time", "Shorter\ntrial", -1),
        ("tool1_normalized_jerk", "Left tool\nnormalised jerk", -1),
        ("tool2_normalized_jerk", "Right tool\nnormalised jerk", -1),
        ("tool1_working_volume", "Left tool compact\nworkspace", -1),
        ("tool2_working_volume", "Right tool compact\nworkspace", -1),
    ]
    norm = _normalised_cohort_means(df, specs)
    if norm.empty:
        return
    metrics = [label for _, label, _ in specs if label in set(norm["metric"])]
    theta = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    fig = plt.figure(figsize=(7.2, 5.9))
    ax = fig.add_subplot(111, polar=True)
    for ds in DATASET_ORDER:
        vals = []
        for metric in metrics:
            value = norm[(norm["dataset"] == ds) & (norm["metric"] == metric)]["value"]
            vals.append(float(value.iloc[0]) if len(value) else np.nan)
        vals = np.asarray(vals, dtype=float)
        vals = np.nan_to_num(vals, nan=np.nanmean(vals) if np.isfinite(vals).any() else 0.0)
        ax.plot(np.r_[theta, theta[0]], np.r_[vals, vals[0]], color=DATASET_COLORS[ds], lw=1.4, label=dataset_label(ds))
        ax.fill(np.r_[theta, theta[0]], np.r_[vals, vals[0]], color=DATASET_COLORS[ds], alpha=0.08)
    ax.set_xticks(theta, metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"])
    ax.tick_params(axis="x", pad=10, labelsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig(fig, "fig16_cohort_radar_summary")


def plot_dataset_metric_matrix(df: pd.DataFrame, proxy: pd.DataFrame) -> None:
    merged = df.copy()
    merged["Coordination-control composite"] = (
        proxy["Coordination-control composite"].to_numpy() if len(proxy) == len(df) else np.nan
    )
    specs = [
        ("Coordination-control composite", "Coordination and control score", 1),
        ("total_time", "Analysed duration", -1),
        ("bimanual_correlation", "Bimanual correlation", 1),
        ("bimanual_lag_s", "Bimanual lag", -1),
        ("bimanual_concurrent_efficiency", "Concurrent efficiency", 1),
        ("simultaneous_motion_ratio", "Simultaneous motion", 1),
        ("combined_idle_ratio", "Idle fraction", -1),
        ("tool_distance_cv", "Tool distance variability", -1),
        ("tool_close_proximity_ratio", "Close proximity", -1),
        ("tool1_normalized_jerk", "Left tool normalised jerk", -1),
        ("tool2_normalized_jerk", "Right tool normalised jerk", -1),
        ("tool1_avg_speed", "Left tool speed", 1),
        ("tool2_avg_speed", "Right tool speed", 1),
        ("tool1_path_length", "Left tool path length", -1),
        ("tool2_path_length", "Right tool path length", -1),
        ("tool1_rotation_per_path", "Left tool rotation economy", -1),
        ("tool2_rotation_per_path", "Right tool rotation economy", -1),
        ("tool1_range_z", "Left tool depth range", -1),
        ("tool2_range_z", "Right tool depth range", -1),
        ("tool1_working_volume", "Left tool working volume", -1),
        ("tool2_working_volume", "Right tool working volume", -1),
        ("tool1_num_speed_peaks", "Left tool speed peaks", -1),
        ("tool2_num_speed_peaks", "Right tool speed peaks", -1),
    ]
    rows = []
    for col, label, direction in specs:
        if col not in merged.columns:
            continue
        vals = pd.to_numeric(merged[col], errors="coerce") * direction
        sd = vals.std(ddof=0)
        z = (vals - vals.mean()) / (sd if sd and np.isfinite(sd) else 1.0)
        for ds in DATASET_ORDER:
            rows.append({"metric": label, "dataset": ds, "z": float(z[merged["dataset"] == ds].mean())})
    mat_df = pd.DataFrame(rows)
    if mat_df.empty:
        return
    metrics = mat_df["metric"].drop_duplicates().to_list()
    mat = np.full((len(DATASET_ORDER), len(metrics)), np.nan)
    for r, metric in enumerate(metrics):
        for c, ds in enumerate(DATASET_ORDER):
            v = mat_df[(mat_df["metric"] == metric) & (mat_df["dataset"] == ds)]["z"]
            mat[c, r] = float(v.iloc[0]) if len(v) else np.nan
    fig = plt.figure(figsize=(10.8, 3.8))
    ax = fig.add_axes([0.08, 0.30, 0.90, 0.38])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), metrics, rotation=45, ha="left")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, pad=1)
    ax.set_yticks(np.arange(len(DATASET_ORDER)), [dataset_label(ds) for ds in DATASET_ORDER])
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            if np.isfinite(mat[r, c]):
                ax.text(c, r, f"{mat[r, c]:.1f}", ha="center", va="center", fontsize=6)
            else:
                ax.text(c, r, "NA", ha="center", va="center", fontsize=6, color="#555555")
    cax = fig.add_axes([0.24, 0.10, 0.52, 0.055])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Cohort mean, oriented z-score")
    save_fig(fig, "fig17_dataset_metric_matrix")


def plot_procedure_motion_disagreement(clusters: pd.DataFrame) -> None:
    """Show how self-reported experience and measured motion bands diverge."""
    if clusters.empty or "metric_cluster" not in clusters:
        return
    d = clusters.dropna(subset=["skill_category", "metric_cluster"]).copy()
    if d.empty:
        return
    matrix = (
        pd.crosstab(d["skill_category"], d["metric_cluster"])
        .reindex(index=SKILL_ORDER, columns=BAND_ORDER)
        .fillna(0)
        .astype(int)
    )
    mismatch_specs = [
        ("Novice in\nupper band", (d["skill_category"] == "novice") & (d["metric_cluster"] == "metric_high"), int((d["skill_category"] == "novice").sum())),
        ("Expert in\nlower band", (d["skill_category"] == "expert") & (d["metric_cluster"] == "metric_low"), int((d["skill_category"] == "expert").sum())),
        ("Novice outside\nlower band", (d["skill_category"] == "novice") & (d["metric_cluster"] != "metric_low"), int((d["skill_category"] == "novice").sum())),
        ("Expert outside\nupper band", (d["skill_category"] == "expert") & (d["metric_cluster"] != "metric_high"), int((d["skill_category"] == "expert").sum())),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    im = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0)
    ax.set_xticks(np.arange(len(BAND_ORDER)), [BAND_SHORT_LABELS[b].replace(" ", "\n") for b in BAND_ORDER])
    ax.set_yticks(np.arange(len(SKILL_ORDER)), [nice(s) for s in SKILL_ORDER])
    ax.set_xlabel("Motion-score band")
    ax.set_ylabel("Procedure-count label")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = int(matrix.iloc[r, c])
            ax.text(c, r, str(value), ha="center", va="center", fontsize=9, weight="bold", color="#111111")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Trials")
    panel_label(ax, "a")

    ax = axes[1]
    counts = np.array([int(mask.sum()) for _, mask, _ in mismatch_specs], dtype=float)
    denoms = np.array([denom if denom else np.nan for _, _, denom in mismatch_specs], dtype=float)
    perc = 100 * counts / denoms
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F58518"]
    x = np.arange(len(mismatch_specs))
    ax.bar(x, counts, color=colors, width=0.72)
    for i, (count, pct) in enumerate(zip(counts, perc)):
        ax.text(i, count + 0.7, f"{int(count)}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [label for label, _, _ in mismatch_specs])
    ax.set_ylabel("Trials")
    ax.set_ylim(0, max(counts) * 1.22 if len(counts) else 1)
    panel_label(ax, "b")
    fig.tight_layout()
    save_fig(fig, "fig18_procedure_motion_disagreement")


def plot_feedback_targets(clusters: pd.DataFrame) -> None:
    """Plot prevalence and magnitude of domain scores below the cohort midpoint."""
    domain_cols = [c for c in OSATS_DOMAINS if c in clusters.columns]
    if clusters.empty or not domain_cols:
        return
    d = clusters.dropna(subset=["metric_cluster", "cohort_label"]).copy()
    domain_order = [
        x
        for x in [*CORE_PERFORMANCE_DOMAINS, "Workspace excursion"]
        if x in domain_cols
    ]
    short = {
        "Bimanual coordination": "Bimanual\ncoordination",
        "Task efficiency": "Task\nefficiency",
        "Instrument motion control": "Instrument\nmotion control",
        "Workspace excursion": "Workspace\nexcursion",
    }
    midpoint = 3.0
    prevalence = pd.DataFrame(index=domain_order, columns=BAND_ORDER, dtype=float)
    deficit = pd.DataFrame(index=domain_order, columns=BAND_ORDER, dtype=float)
    for domain in domain_order:
        for band in BAND_ORDER:
            values = d.loc[d["metric_cluster"] == band, domain].dropna()
            prevalence.loc[domain, band] = 100 * (values < midpoint).mean() if len(values) else np.nan
            deficit.loc[domain, band] = np.maximum(0, midpoint - values).mean() if len(values) else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.85), gridspec_kw={"width_ratios": [0.95, 1.35]})
    ax = axes[0]
    im = ax.imshow(prevalence.to_numpy(float), vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(BAND_ORDER)), [BAND_SHORT_LABELS[b] for b in BAND_ORDER])
    ax.set_yticks(range(len(domain_order)), [short[d].replace("\n", " ") for d in domain_order])
    for r in range(len(domain_order)):
        for c in range(len(BAND_ORDER)):
            value = prevalence.iloc[r, c]
            ax.text(c, r, "NA" if not np.isfinite(value) else f"{value:.0f}%", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.20, fraction=0.08)
    cbar.set_label("Trials below score 3.0")
    panel_label(ax, "a")

    ax = axes[1]
    x = np.arange(len(domain_order))
    width = 0.24
    for i, band in enumerate(BAND_ORDER):
        ax.bar(
            x + (i - 1) * width,
            deficit[band].to_numpy(float),
            width=width,
            color=CLUSTER_COLORS[band],
            label=BAND_SHORT_LABELS[band],
        )
    ax.set_xticks(x, [short[domain] for domain in domain_order])
    ax.set_ylabel("Mean shortfall below score 3.0")
    ax.legend(frameon=False, fontsize=6.6, loc="upper right")
    panel_label(ax, "b")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.22, wspace=0.38)
    save_fig(fig, "fig19_feedback_targets")


def _phase_trials_with_clusters(trials: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    phase = trials.copy()
    phase["trial_key"] = [phase_trial_key(ds, ts) for ds, ts in zip(phase["dataset"], phase["trial_short"])]
    band_col = "metric_cluster"
    cluster_lookup = (
        clusters[["dataset", "trial_name", band_col, "cohort_label"]]
        .rename(columns={band_col: "metric_cluster"})
        .drop_duplicates(["dataset", "trial_name"])
    )
    phase = phase.merge(
        cluster_lookup,
        left_on=["dataset", "trial_short"],
        right_on=["dataset", "trial_name"],
        how="left",
        validate="one_to_one",
    )
    identifier_count = phase.groupby("trial_key")["trial_key"].transform("size")
    return phase.loc[identifier_count.eq(1)].copy()


def _phase_cycles_with_clusters(cycles: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    cyc = cycles.copy()
    cyc["trial_key"] = [
        phase_trial_key(ds, trial_short)
        for ds, trial_short in zip(cyc["dataset"], cyc["trial_short"])
    ]
    band_col = "metric_cluster"
    cluster_lookup = (
        clusters[["dataset", "trial_name", band_col, "cohort_label"]]
        .rename(columns={band_col: "metric_cluster"})
        .drop_duplicates(["dataset", "trial_name"])
    )
    cyc = cyc.merge(
        cluster_lookup,
        left_on=["dataset", "trial_short"],
        right_on=["dataset", "trial_name"],
        how="left",
        validate="many_to_one",
    )
    identifier_recordings = (
        cyc[["trial_key", "trial_id"]].drop_duplicates().groupby("trial_key")["trial_id"].transform("size")
    )
    unique_keys = set(
        cyc[["trial_key", "trial_id"]]
        .drop_duplicates()
        .loc[identifier_recordings.eq(1), "trial_key"]
    )
    return cyc.loc[cyc["trial_key"].isin(unique_keys)].copy()


def plot_phase_bottlenecks(trials: pd.DataFrame, cycles: pd.DataFrame, clusters: pd.DataFrame) -> None:
    """Compare phase timing across descriptive motion-score bands."""
    if clusters.empty:
        return
    phase = _phase_trials_with_clusters(trials, clusters).dropna(subset=["metric_cluster"])
    cyc = _phase_cycles_with_clusters(cycles, clusters).dropna(subset=["metric_cluster"])
    if phase.empty or cyc.empty:
        return
    trial_cycle = (
        cyc.groupby(["metric_cluster", "trial_key"], as_index=False)
        .agg(
            mean_cycle_duration_s=("duration_s", "mean"),
            mean_transitions=("n_transitions", "mean"),
        )
    )
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(9.4, 3.35),
        gridspec_kw={"width_ratios": [0.9, 1.2, 0.9, 0.9]},
    )

    ax = axes[0]
    means = []
    los = []
    his = []
    for band in BAND_ORDER:
        vals = trial_cycle.loc[
            trial_cycle["metric_cluster"] == band,
            "mean_cycle_duration_s",
        ].dropna()
        mean = vals.mean()
        lo, hi = bootstrap_ci(vals)
        means.append(mean)
        los.append(mean - lo)
        his.append(hi - mean)
    x = np.arange(len(BAND_ORDER))
    ax.bar(x, means, color=[CLUSTER_COLORS[b] for b in BAND_ORDER], width=0.72)
    ax.errorbar(x, means, yerr=[los, his], color="#222222", fmt="none", capsize=3, lw=0.8)
    ax.set_xticks(x, [BAND_SHORT_LABELS[b].replace(" ", "\n") for b in BAND_ORDER])
    ax.set_ylabel("Cycle duration (s)")
    panel_label(ax, "a")

    ax = axes[1]
    phase_cols = ["frac_reach", "frac_grasp", "frac_transfer", "frac_place", "frac_nudge", "frac_idle", "frac_dropped"]
    means_df = phase.groupby("metric_cluster")[phase_cols].mean().reindex(BAND_ORDER).fillna(0)
    left = np.zeros(len(BAND_ORDER))
    y = np.arange(len(BAND_ORDER))
    for col in phase_cols:
        phase_name = col.replace("frac_", "")
        vals = means_df[col].to_numpy()
        ax.barh(y, vals, left=left, color=PHASE_COLORS[phase_name], label=nice(phase_name), height=0.68)
        left += vals
    ax.set_yticks(y, [BAND_SHORT_LABELS[b] for b in BAND_ORDER])
    ax.set_xlabel("Mean fraction of corrected trial time")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    panel_label(ax, "b")

    ax = axes[2]
    trans_means = (
        trial_cycle.groupby("metric_cluster")["mean_transitions"]
        .mean()
        .reindex(BAND_ORDER)
    )
    trans_los = []
    trans_his = []
    for band in BAND_ORDER:
        vals = trial_cycle.loc[
            trial_cycle["metric_cluster"] == band,
            "mean_transitions",
        ].dropna()
        mean = vals.mean()
        lo, hi = bootstrap_ci(vals)
        trans_los.append(mean - lo)
        trans_his.append(hi - mean)
    ax.bar(x, trans_means, color=[CLUSTER_COLORS[b] for b in BAND_ORDER], width=0.72)
    ax.errorbar(x, trans_means, yerr=[trans_los, trans_his], color="#222222", fmt="none", capsize=3, lw=0.8)
    ax.set_xticks(x, [BAND_SHORT_LABELS[b].replace(" ", "\n") for b in BAND_ORDER])
    ax.set_ylabel("Phase transitions per cycle")
    panel_label(ax, "c")

    ax = axes[3]
    trial_drops = (
        cyc.groupby(["metric_cluster", "trial_key"], as_index=False)
        .agg(
            n_cycles=("cycle_index", "count"),
            n_drop_events=("n_drop_events", "sum"),
        )
    )
    trial_drops["drop_events_per_100_cycles"] = (
        100 * trial_drops["n_drop_events"] / trial_drops["n_cycles"]
    )
    drop_means = []
    drop_los = []
    drop_his = []
    for band in BAND_ORDER:
        values = trial_drops.loc[
            trial_drops["metric_cluster"] == band,
            "drop_events_per_100_cycles",
        ].dropna()
        mean = values.mean()
        lo, hi = bootstrap_ci(values)
        drop_means.append(mean)
        drop_los.append(mean - lo)
        drop_his.append(hi - mean)
    ax.bar(x, drop_means, color=[CLUSTER_COLORS[b] for b in BAND_ORDER], width=0.72)
    ax.errorbar(
        x,
        drop_means,
        yerr=[drop_los, drop_his],
        color="#222222",
        fmt="none",
        capsize=3,
        lw=0.8,
    )
    ax.set_xticks(x, [BAND_SHORT_LABELS[b].replace(" ", "\n") for b in BAND_ORDER])
    ax.set_ylabel("Drop episodes per 100 cycles")
    panel_label(ax, "d")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=7, fontsize=6.4, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.90, bottom=0.24, wspace=0.43)
    save_fig(fig, "fig20_phase_bottlenecks")


def plot_cohort_effect_forest(df: pd.DataFrame) -> None:
    """Cohort-specific expert-versus-novice effects for clinically important metrics."""
    identifier_counts = df.groupby("trial_key")["trial_key"].transform("size")
    df = df.loc[identifier_counts.eq(1)].copy()
    specs = [
        ("tool2_avg_speed", "Right tool speed", 1),
        ("tool1_avg_speed", "Left tool speed", 1),
        ("bimanual_correlation", "Bimanual correlation", 1),
        ("tool2_num_speed_peaks", "Right tool stop-start peaks", -1),
        ("tool1_num_speed_peaks", "Left tool stop-start peaks", -1),
        ("total_time", "Analysed duration", -1),
        ("tool2_range_z", "Right tool depth range", -1),
    ]
    rows = []
    rng = np.random.default_rng(20260618)
    for col, label, direction in specs:
        if col not in df.columns:
            continue
        for ds in DATASET_ORDER:
            g = df[df["dataset"] == ds]
            nov = (pd.to_numeric(g.loc[g["skill_category"] == "novice", col], errors="coerce") * direction).dropna().to_numpy()
            exp = (pd.to_numeric(g.loc[g["skill_category"] == "expert", col], errors="coerce") * direction).dropna().to_numpy()
            if len(nov) == 0 or len(exp) == 0:
                continue
            delta = cliffs_delta(exp, nov)
            boot = []
            if len(nov) > 1 and len(exp) > 1:
                for _ in range(1000):
                    boot.append(cliffs_delta(rng.choice(exp, len(exp), replace=True), rng.choice(nov, len(nov), replace=True)))
                lo, hi = np.percentile(boot, [2.5, 97.5])
            else:
                lo, hi = np.nan, np.nan
            rows.append({"feature": label, "dataset": ds, "delta": delta, "lo": lo, "hi": hi})
    res = pd.DataFrame(rows)
    if res.empty:
        return
    res.to_csv(OUT / "cohort_specific_effects.csv", index=False)

    features = [label for _, label, _ in specs if label in set(res["feature"])]
    y_base = np.arange(len(features))
    offsets = {"BAPES2024": -0.22, "6DOF2023": 0.0, "7DOF2024": 0.22}
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for ds in DATASET_ORDER:
        sub = res[res["dataset"] == ds]
        ys = np.array([features.index(f) for f in sub["feature"]], dtype=float) + offsets[ds]
        xerr = np.vstack([(sub["delta"] - sub["lo"]).clip(lower=0), (sub["hi"] - sub["delta"]).clip(lower=0)])
        ax.errorbar(
            sub["delta"],
            ys,
            xerr=xerr,
            fmt="o",
            ms=4,
            lw=0.8,
            capsize=2.5,
            color=DATASET_COLORS[ds],
            label=dataset_label(ds),
            alpha=0.92,
        )
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y_base, features)
    ax.invert_yaxis()
    ax.set_xlabel("Oriented Cliff's delta for expert minus novice")
    ax.set_xlim(-1.05, 1.05)
    ax.text(0.01, 1.015, "Novice trials more favourable", transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
    ax.text(0.99, 1.015, "Expert trials more favourable", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_fig(fig, "fig21_cohort_effect_forest")


def plot_time_synchrony_quadrants(df: pd.DataFrame, clusters: pd.DataFrame) -> None:
    """Contrast completion time with coordination to show why time alone is incomplete."""
    if "total_time" not in df.columns or "bimanual_correlation" not in df.columns:
        return
    identifier_counts = df.groupby("trial_key")["trial_key"].transform("size")
    d = df.loc[identifier_counts.eq(1)].copy()
    correlation_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(8.7, 3.1), sharex=False, sharey=True)
    for ax, ds, label in zip(axes, DATASET_ORDER, "abc"):
        g = d[d["dataset"] == ds].dropna(subset=["total_time", "bimanual_correlation"])
        if g.empty:
            ax.axis("off")
            continue
        for skill in SKILL_ORDER:
            sub = g[g["skill_category"] == skill]
            ax.scatter(
                sub["total_time"],
                sub["bimanual_correlation"],
                s=22,
                alpha=0.82,
                color=SKILL_COLORS[skill],
                label=nice(skill),
            )
        xmed = g["total_time"].median()
        ymed = g["bimanual_correlation"].median()
        ax.axvline(xmed, color="#555555", lw=0.8, ls=":")
        ax.axhline(ymed, color="#555555", lw=0.8, ls=":")
        rho, p = stats.spearmanr(
            g["total_time"],
            g["bimanual_correlation"],
            nan_policy="omit",
        )
        ci_low, ci_high = bootstrap_spearman_ci(
            g["total_time"],
            g["bimanual_correlation"],
        )
        correlation_rows.append(
            {
                "dataset": ds,
                "n": int(len(g)),
                "spearman_rho": float(rho),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p": float(p),
            }
        )
        ax.text(
            0.04,
            0.96,
            f"$n={len(g)}$\n$\\rho={rho:.2f}$ [{ci_low:.2f}, {ci_high:.2f}]",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.4),
        )
        ax.text(
            0.04,
            0.06,
            "Shorter",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.2,
            color="#555555",
        )
        ax.text(
            0.96,
            0.06,
            "Longer",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.2,
            color="#555555",
        )
        ax.set_title(dataset_label(ds), fontsize=8)
        ax.set_xlabel("Analysed duration (s)")
        if ax is axes[0]:
            ax.set_ylabel("Bimanual correlation")
        panel_label(ax, label)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.25, wspace=0.24)
    save_fig(fig, "fig22_time_synchrony_quadrants")
    pd.DataFrame(correlation_rows).to_csv(
        OUT / "coordination_control_time_correlation.csv",
        index=False,
    )


def plot_continuous_performance_validation(
    proxy: pd.DataFrame,
    clusters: pd.DataFrame,
) -> None:
    """Show construction-excluded relationships without depending on cut points."""
    d = proxy.merge(
        clusters[["dataset", "trial_name", "metric_cluster"]].drop_duplicates(
            ["dataset", "trial_name"]
        ),
        on=["dataset", "trial_name"],
        how="left",
        validate="one_to_one",
    )
    identifier_counts = d.groupby("trial_key")["trial_key"].transform("size")
    d = d.loc[identifier_counts.eq(1)].copy()
    outcomes = [
        ("Task efficiency", "Task efficiency score (1-5)"),
        ("Analysed duration (s)", "Analysed duration (s)"),
    ]
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(8.9, 5.8), sharex="col")
    for r, (outcome, ylabel) in enumerate(outcomes):
        for c, (ax, ds, label) in enumerate(zip(axes[r], DATASET_ORDER, "abcdef"[r * 3 : r * 3 + 3])):
            g = d[d["dataset"] == ds].dropna(
                subset=["Coordination-control composite", outcome, "metric_cluster"]
            )
            for band in BAND_ORDER:
                sub = g[g["metric_cluster"] == band]
                ax.scatter(
                    sub["Coordination-control composite"],
                    sub[outcome],
                    s=24,
                    alpha=0.78,
                    color=CLUSTER_COLORS[band],
                    edgecolor="white",
                    linewidth=0.25,
                    label=BAND_SHORT_LABELS[band],
                )
            if len(g) >= 3:
                x = g["Coordination-control composite"].to_numpy(float)
                y = g[outcome].to_numpy(float)
                slope, intercept = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 100)
                ax.plot(xline, intercept + slope * xline, color="#333333", lw=1.0)
                rho, p_value = stats.spearmanr(x, y)
                ci_low, ci_high = bootstrap_spearman_ci(x, y)
                rows.append(
                    {
                        "dataset": ds,
                        "outcome": outcome,
                        "n": len(g),
                        "spearman_rho": rho,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p": p_value,
                    }
                )
                ax.text(
                    0.04,
                    0.96,
                    f"$n={len(g)}$\n$\\rho={rho:.2f}$ [{ci_low:.2f}, {ci_high:.2f}]",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.4),
                )
            if r == 0:
                ax.set_title(dataset_label(ds), fontsize=8)
            if r == len(outcomes) - 1:
                ax.set_xlabel("Coordination and control score (1-5)")
            if c == 0:
                ax.set_ylabel(ylabel)
            if outcome == "Task efficiency":
                ax.set_ylim(1, 5)
            panel_label(ax, label)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["q"] = benjamini_hochberg(result["p"])
        result.to_csv(OUT / "continuous_performance_validation.csv", index=False)
        result["dataset"] = pd.Categorical(
            result["dataset"], categories=DATASET_ORDER, ordered=True
        )
        result = result.sort_values(["dataset", "outcome"])
        table_rows = []
        for row in result.itertuples(index=False):
            q_text = "$<0.001$" if row.q < 0.001 else f"{row.q:.3f}"
            table_rows.append(
                f"{dataset_label(row.dataset)} & {row.outcome.replace(' (s)', '')} & "
                f"{int(row.n)} & {row.spearman_rho:.2f} "
                f"[{row.ci_low:.2f}, {row.ci_high:.2f}] & {q_text} \\\\"
            )
        (TAB / "continuous_performance_validation.tex").write_text(
            "\n".join(
                [
                    "\\begin{tabular}{llrrr}",
                    "\\toprule",
                    "Cohort & Contextual outcome & $n$ & Spearman $\\rho$ [95\\% CI] & FDR $q$ \\\\",
                    "\\midrule",
                    *table_rows,
                    "\\bottomrule",
                    "\\end{tabular}",
                ]
            )
        )
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.16, hspace=0.28, wspace=0.22)
    save_fig(fig, "fig23_continuous_performance_validation")


def write_tables(
    df: pd.DataFrame,
    phase_trials: pd.DataFrame,
    phase_cycles: pd.DataFrame,
    feature_stats: pd.DataFrame,
    adjusted_models: pd.DataFrame,
    phase_kinematics: pd.DataFrame,
    classification: dict[str, object],
    proxy: pd.DataFrame,
    clusters: pd.DataFrame,
    cluster_summary: dict[str, object],
    metric_band_classification: dict[str, object],
) -> None:
    dataset_rows = []
    for ds in DATASET_ORDER:
        g = df[df["dataset"] == ds]
        skills = g["skill_category"].value_counts()
        hands = g["handedness"].value_counts()
        context = "BAPES training event" if ds == "BAPES2024" else "Urology"
        dataset_rows.append(
            f"{dataset_label(ds)} & {context} & {g['dof'].iloc[0]} & {len(g)} & "
            f"{g['trial_key'].nunique()} & "
            f"{int(skills.get('novice', 0))}/{int(skills.get('intermediate', 0))}/{int(skills.get('expert', 0))} & "
            f"{int(hands.get('right', 0))}/{int(hands.get('left', 0))}/{int(hands.get('mixed', 0))} \\\\"
        )
    (TAB / "dataset_characteristics.tex").write_text(
        "\n".join(
            [
                "\\begin{tabular}{lllrrrr}",
                "\\toprule",
                "Cohort & Collection context & Sensing & Recordings & Study IDs & Experience N/I/E & Hand R/L/M \\\\",
                "\\midrule",
                *dataset_rows,
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
    )

    identifier_counts = df.groupby("trial_key")["trial_key"].transform("size")
    independent_keys = set(df.loc[identifier_counts.eq(1), "trial_key"])
    phase_inferential = phase_trials[phase_trials["trial_key"].isin(independent_keys)]
    analysis_unit_rows = [
        "Public Zenodo version 1.0 & 37 & 37 & Currently released subset \\\\",
        "Earlier LASK report & 114 & -- & Published collection summary \\\\",
        f"Dataset inventory & {len(df)} & {df['trial_key'].nunique()} & All available recordings \\\\",
        f"Primary inferential set & {int(identifier_counts.eq(1).sum())} & "
        f"{df.loc[identifier_counts.eq(1), 'trial_key'].nunique()} & "
        "Identifiers represented by one recording \\\\",
        f"Annotated phase inventory & {len(phase_trials)} & {phase_trials['trial_key'].nunique()} & "
        "All densely annotated recordings \\\\",
        f"Phase inferential set & {len(phase_inferential)} & "
        f"{phase_inferential['trial_key'].nunique()} & "
        "Identifiers represented once in full inventory \\\\",
    ]
    (TAB / "analysis_units.tex").write_text(
        "\n".join(
            [
                "\\begin{tabular}{lrrL{5.0cm}}",
                "\\toprule",
                "Analysis set & Recordings & Study IDs & Use \\\\",
                "\\midrule",
                *analysis_unit_rows,
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
    )

    bimanual = feature_stats.loc[
        feature_stats["feature"].eq("bimanual_correlation")
    ]
    if not bimanual.empty:
        row = bimanual.iloc[0]
        (TAB / "bimanual_meta_sensitivity.tex").write_text(
            "\n".join(
                [
                    "\\begin{tabular}{lllll}",
                    "\\toprule",
                    "Model & Spearman $\\rho$ & 95\\% CI & Evidence value & Heterogeneity \\\\",
                    "\\midrule",
                    "Fixed effect (primary) & "
                    f"{row['spearman_rho_procedures']:.2f} & "
                    f"[{row['spearman_ci_low']:.2f}, {row['spearman_ci_high']:.2f}] & "
                    f"FDR $q={row['spearman_q']:.3f}$ & "
                    f"$I^2={row['spearman_meta_i2']:.1f}\\%$ \\\\",
                    "Random effects (sensitivity) & "
                    f"{row['spearman_random_rho']:.2f} & "
                    f"[{row['spearman_random_ci_low']:.2f}, {row['spearman_random_ci_high']:.2f}] & "
                    f"Unadjusted $p={row['spearman_random_p']:.3f}$ & "
                    f"$\\tau^2={row['spearman_random_tau2']:.3f}$ \\\\",
                    "\\bottomrule",
                    "\\end{tabular}",
                ]
            )
        )

    phase_rows = []
    for ds in DATASET_ORDER:
        g = phase_trials[phase_trials["dataset"] == ds]
        skills = g["skill_category"].value_counts()
        phase_rows.append(
            f"{dataset_label(ds)} & {len(g)} & {g['trial_key'].nunique()} & "
            f"{int(skills.get('novice', 0))}/{int(skills.get('intermediate', 0))}/{int(skills.get('expert', 0))} & "
            f"{int(g['n_frames'].sum()):,} & {int(g['n_cycles'].sum())} & {g['mean_cycle_s'].mean():.1f} \\\\"
        )
    (TAB / "phase_subset.tex").write_text(
        "\n".join(
            [
                "\\begin{tabular}{lrrrrrr}",
                "\\toprule",
                "Dataset & Recordings & IDs & Procedure N/I/E & Frames & Cycles & Mean cycle (s) \\\\",
                "\\midrule",
                *phase_rows,
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
    )

    annotated_names = set(
        zip(
            phase_trials["dataset"].astype(str),
            phase_trials["trial_short"].astype(str),
        )
    )
    selection = df[["dataset", "trial_name", "trial_key", "total_time"]].copy()
    selection["phase_annotated"] = [
        (str(dataset), str(name)) in annotated_names
        for dataset, name in zip(selection["dataset"], selection["trial_name"])
    ]
    selection = selection.merge(
        clusters[
            ["dataset", "trial_name", "Coordination-control composite"]
        ].drop_duplicates(["dataset", "trial_name"]),
        on=["dataset", "trial_name"],
        how="left",
        validate="one_to_one",
    )
    selection_counts = selection.groupby("trial_key")["trial_key"].transform("size")
    selection = selection.loc[selection_counts.eq(1)].copy()
    selection_rows = []
    for ds in DATASET_ORDER:
        cohort = selection[selection["dataset"] == ds]
        annotated = cohort[cohort["phase_annotated"]]
        other = cohort[~cohort["phase_annotated"]]
        selection_rows.append(
            f"{dataset_label(ds)} & {len(annotated)}/{len(cohort)} & "
            f"{annotated['total_time'].median():.1f}/{other['total_time'].median():.1f} & "
            f"{annotated['Coordination-control composite'].median():.2f}/"
            f"{other['Coordination-control composite'].median():.2f} \\\\"
        )
    (TAB / "phase_subset_selection.tex").write_text(
        "\n".join(
            [
                "\\begin{tabular}{lrrr}",
                "\\toprule",
                "Cohort & Annotated/available & Median time annotated/not (s) & "
                "Median coordination and control score annotated/not \\\\",
                "\\midrule",
                *selection_rows,
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
    )

    rows = []
    feature_rows = feature_stats[feature_stats["n_nonmissing"] >= 20].copy()
    feature_rows = feature_rows.loc[
        feature_rows["cliffs_delta_novice_expert"]
        .abs()
        .sort_values(ascending=False)
        .index
    ].head(12)
    for _, r in feature_rows.iterrows():
        arrow, interpretation = FEATURE_DIRECTIONS.get(r["feature"], ("", ""))
        higher_is_better = "\\uparrow" in arrow
        effect_orientation = -1 if higher_is_better else 1
        rho_orientation = 1 if higher_is_better else -1
        oriented_delta = effect_orientation * r["cliffs_delta_novice_expert"]
        if effect_orientation > 0:
            oriented_lo, oriented_hi = r["delta_ci_low"], r["delta_ci_high"]
        else:
            oriented_lo, oriented_hi = -r["delta_ci_high"], -r["delta_ci_low"]
        kruskal_q = "$<0.001$" if r["kruskal_q"] < 0.001 else f"{r['kruskal_q']:.3f}"
        spearman_q = "$<0.001$" if r["spearman_q"] < 0.001 else f"{r['spearman_q']:.3f}"
        if rho_orientation > 0:
            rho_lo, rho_hi = r["spearman_ci_low"], r["spearman_ci_high"]
        else:
            rho_lo, rho_hi = -r["spearman_ci_high"], -r["spearman_ci_low"]
        rows.append(
            f"{r['label']} ({arrow}) & {interpretation} & "
            f"{oriented_delta:.2f} [{oriented_lo:.2f}, {oriented_hi:.2f}] & "
            f"{kruskal_q} & {rho_orientation * r['spearman_rho_procedures']:.2f} "
            f"[{rho_lo:.2f}, {rho_hi:.2f}; {spearman_q}] \\\\"
        )
    (TAB / "feature_effects.tex").write_text(
        "\n".join(
            [
                        "\\begin{tabular}{L{0.17\\linewidth}L{0.27\\linewidth}L{0.18\\linewidth}L{0.11\\linewidth}L{0.16\\linewidth}}",
                "\\toprule",
                "Feature & Prespecified interpretation & Oriented Cliff $\\delta$ [95\\% CI] & "
                "Three-group $q$ & Oriented meta-$\\rho$ [95\\% CI; $q$] \\\\",
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
    )

    if not adjusted_models.empty:
        rows = []
        for _, r in adjusted_models.iterrows():
            q_text = (
                "$<0.001$"
                if r["procedure_q"] < 0.001
                else f"{r['procedure_q']:.3f}"
            )
            rows.append(
                f"{r['label']} & {r['beta_log10_procedures']:.2f} & "
                f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] & "
                f"{q_text} & "
                f"{r['r2_procedure_only']:.2f} & {r['r2_cohort_only']:.2f} & "
                f"{r['r2_full']:.2f} & {int(r['n'])} \\\\"
            )
        (TAB / "cohort_adjusted_models.tex").write_text(
            "\n".join(
                [
                    "\\begin{tabular}{lrrrrrrr}",
                    "\\toprule",
                    "Outcome & $\\beta_{\\log_{10}\\mathrm{proc}}$ & 95\\% CI & FDR $q$ & $R^2$ proc. & $R^2$ cohort & Full $R^2$ & $n$ \\\\",
                    "\\midrule",
                    *rows,
                    "\\bottomrule",
                    "\\end{tabular}",
                ]
            )
        )

    if classification.get("available"):
        cv = classification["stratified_group_5fold"]
        lodo = classification["leave_one_dataset_out"]
        model_labels = {
            "time_only_logistic_regression": "Duration-only logistic regression",
            "time_only_random_forest": "Duration-only random forest",
            "all_features_logistic_regression": "All-feature logistic regression",
            "all_features_random_forest": "All-feature random forest",
        }
        lines = [
            "\\begin{tabular}{L{0.35\\linewidth}L{0.22\\linewidth}L{0.31\\linewidth}}",
            "\\toprule",
            "Model & 5-fold balanced accuracy, mean (SD) & Leave-one-cohort-out balanced accuracy, mean [range] \\\\",
            "\\midrule",
        ]
        for model, vals in cv.items():
            lodo_scores = [r["balanced_accuracy"] for r in lodo[model]]
            mean_lodo = np.mean(lodo_scores)
            lines.append(
                f"{model_labels.get(model, model.replace('_', ' ').title())} & "
                f"{vals['mean_balanced_accuracy']:.2f} ({vals['sd']:.2f}) & "
                f"{mean_lodo:.2f} [{np.min(lodo_scores):.2f}, {np.max(lodo_scores):.2f}] \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}"]
        (TAB / "classification.tex").write_text("\n".join(lines))

    if metric_band_classification.get("available"):
        comparison_rows = metric_band_classification.get("comparison_rows", [])
        target_order = {
            "motion_defined": 0,
            "procedure_fixed": 1,
            "procedure_tertiles": 2,
            "procedure_kmeans": 3,
        }
        model_order = {"logistic_regression": 0, "random_forest": 1}
        comparison_rows = sorted(
            comparison_rows,
            key=lambda r: (target_order.get(r.get("target"), 99), model_order.get(r.get("model"), 99)),
        )
        lines = [
            "\\begin{tabular}{L{0.18\\linewidth}L{0.18\\linewidth}L{0.27\\linewidth}L{0.28\\linewidth}}",
            "\\toprule",
            "Target rule & Model & Group cross-validation, mean (SD) & Cross-cohort rule recovery, mean [range] \\\\",
            "\\midrule",
        ]
        for r in comparison_rows:
            lines.append(
                f"{r['target_label']} & {r['model_label']} & "
                f"{r['stratified_cv_balanced_accuracy']:.2f} "
                f"({r['stratified_cv_sd']:.2f}) & "
                f"{r['leave_one_dataset_out_balanced_accuracy']:.2f} "
                f"[{r['leave_one_dataset_out_min']:.2f}, "
                f"{r['leave_one_dataset_out_max']:.2f}] \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}"]
        (TAB / "metric_band_classification.tex").write_text("\n".join(lines))

    if not clusters.empty and "metric_cluster" in clusters:
        d = clusters.dropna(subset=["skill_category", "metric_cluster"]).copy()
        denom_novice = int((d["skill_category"] == "novice").sum())
        denom_expert = int((d["skill_category"] == "expert").sum())
        disagreement_specs = [
            ("Novice procedure group in upper band", (d["skill_category"] == "novice") & (d["metric_cluster"] == "metric_high"), denom_novice),
            ("Expert procedure group in lower band", (d["skill_category"] == "expert") & (d["metric_cluster"] == "metric_low"), denom_expert),
            ("Novice procedure group outside lower band", (d["skill_category"] == "novice") & (d["metric_cluster"] != "metric_low"), denom_novice),
            ("Expert procedure group outside upper band", (d["skill_category"] == "expert") & (d["metric_cluster"] != "metric_high"), denom_expert),
        ]
        rows = []
        for label, mask, denom in disagreement_specs:
            count = int(mask.sum())
            pct = 100 * count / denom if denom else np.nan
            rows.append(f"{label} & {count}/{denom} & {pct:.1f}\\% \\\\")
        (TAB / "performance_disagreement.tex").write_text(
            "\n".join(
                [
                    "\\begin{tabular}{lrr}",
                    "\\toprule",
                    "Comparison & Count & Percentage \\\\",
                    "\\midrule",
                    *rows,
                    "\\bottomrule",
                    "\\end{tabular}",
                ]
            )
        )

    if not clusters.empty and cluster_summary.get("available"):
        rows = []
        order = {cl: i for i, cl in enumerate(BAND_ORDER)}
        thresholds = cluster_summary.get("metric_band_thresholds", [np.nan, np.nan])
        threshold_rules = {
            "metric_low": f"$< {thresholds[0]:.2f}$",
            "metric_middle": f"${thresholds[0]:.2f}$ to $< {thresholds[1]:.2f}$",
            "metric_high": f"$\\geq {thresholds[1]:.2f}$",
        }
        for cl in BAND_ORDER:
            g = clusters[clusters["metric_cluster"] == cl]
            if g.empty:
                continue
            rows.append(
                (
                    order[cl],
                    f"{BAND_LABELS[cl]} & {threshold_rules[cl]} & {len(g)} & "
                    f"{g['Coordination-control composite'].mean():.2f} & {g['total_procedures'].median():.0f} "
                    f"[{g['total_procedures'].quantile(0.25):.0f}, {g['total_procedures'].quantile(0.75):.0f}] & "
                    f"{int((g['skill_category']=='novice').sum())}/{int((g['skill_category']=='intermediate').sum())}/{int((g['skill_category']=='expert').sum())} \\\\"
                )
            )
        rows = [r for _, r in sorted(rows)]
        (TAB / "kmeans_clusters.tex").write_text(
            "\n".join(
                [
                    "\\begin{tabular}{llrrrr}",
                    "\\toprule",
                    "Motion-score band & Score rule & Trials & Mean score & Procedure median [IQR] & Procedure N/I/E \\\\",
                    "\\midrule",
                    *rows,
                    "\\bottomrule",
                    "\\end{tabular}",
                ]
            )
        )
        held_out_path = OUT / "held_out_band_statistics.csv"
        if held_out_path.exists():
            held_out = pd.read_csv(held_out_path)
            held_out_rows = []
            for _, row in held_out.iterrows():
                if row["outcome"] == "Workspace excursion":
                    outcome_label = "Workspace compactness (descriptive)"
                    arrow = ""
                else:
                    outcome_label = str(row["outcome"])
                    arrow = " ($\\uparrow$)" if row["direction"] == "higher" else " ($\\downarrow$)"
                q_text = "$<0.001$" if row["upper_lower_q"] < 0.001 else f"{row['upper_lower_q']:.3f}"
                held_out_rows.append(
                    f"{outcome_label}{arrow} & "
                    f"{row['median_lower']:.2f}/{row['median_middle']:.2f}/{row['median_upper']:.2f} & "
                    f"{row['oriented_cliffs_delta_upper_vs_lower']:.2f} "
                    f"[{row['delta_ci_low']:.2f}, {row['delta_ci_high']:.2f}] & "
                    f"{q_text} & "
                    f"{int(row['n_lower'])}/{int(row['n_middle'])}/{int(row['n_upper'])} \\\\"
                )
            (TAB / "held_out_band_checks.tex").write_text(
                "\n".join(
                    [
                        "\\begin{tabular}{lrrrr}",
                        "\\toprule",
                        "Contextual outcome & Lower/middle/upper median & "
                        "Oriented Cliff $\\delta$ [95\\% CI] & FDR $q$ & $n$ L/M/U \\\\",
                        "\\midrule",
                        *held_out_rows,
                        "\\bottomrule",
                        "\\end{tabular}",
                    ]
                )
            )
        validation = pd.read_csv(OUT / "kmeans_validation.csv") if (OUT / "kmeans_validation.csv").exists() else pd.DataFrame()
        if not validation.empty:
            val_rows = [
                f"{int(r['k'])} & {r['silhouette']:.2f} & {r['davies_bouldin']:.2f} & {r['calinski_harabasz']:.1f} & {r['inertia']:.1f} \\\\"
                for _, r in validation.iterrows()
            ]
            (TAB / "kmeans_validation.tex").write_text(
                "\n".join(
                    [
                        "\\begin{tabular}{rrrrr}",
                        "\\toprule",
                        "$k$ & Silhouette $\\uparrow$ & Davies--Bouldin $\\downarrow$ & Calinski--Harabasz $\\uparrow$ & Inertia $\\downarrow$ \\\\",
                        "\\midrule",
                        *val_rows,
                        "\\bottomrule",
                        "\\end{tabular}",
                    ]
                )
            )
        sensitivity = cluster_summary.get("sensitivity", {})
        sensitivity_rows = []
        complete = sensitivity.get("complete_case", {})
        complete_cuts = complete.get("thresholds", [np.nan, np.nan])
        sensitivity_rows.append(
            "Complete raw-feature cases & "
            f"$n={int(complete.get('n_trials', 0))}$ & "
            f"{complete_cuts[0]:.2f}, {complete_cuts[1]:.2f} & "
            f"{complete.get('assignment_adjusted_rand_index', np.nan):.2f} \\\\"
        )
        perturb = sensitivity.get("domain_weight_perturbation", {})
        sensitivity_rows.append(
            "Domain weights varied by 25\\% & "
            f"{int(perturb.get('n_repetitions', 0))} repetitions & -- & "
            f"{perturb.get('adjusted_rand_index_median', np.nan):.2f} "
            f"[{perturb.get('adjusted_rand_index_ci_low', np.nan):.2f}, "
            f"{perturb.get('adjusted_rand_index_ci_high', np.nan):.2f}] \\\\"
        )
        for row in sensitivity.get("leave_one_domain_out", []):
            cuts = row.get("thresholds", [np.nan, np.nan])
            sensitivity_rows.append(
                f"Omit {row['omitted_domain']} & Remaining one-domain score & "
                f"{cuts[0]:.2f}, {cuts[1]:.2f} & {row['adjusted_rand_index']:.2f} \\\\"
            )
        for row in sensitivity.get("leave_one_cohort_out", []):
            cuts = row.get("thresholds", [np.nan, np.nan])
            sensitivity_rows.append(
                f"Fit without {dataset_label(row['held_out_dataset'])} & Excluded cohort & "
                f"{cuts[0]:.2f}, {cuts[1]:.2f} & {row['held_out_adjusted_rand_index']:.2f} \\\\"
            )
        (TAB / "band_sensitivity.tex").write_text(
            "\n".join(
                [
                    "\\begin{tabular}{llll}",
                    "\\toprule",
                    "Sensitivity analysis & Data or perturbation & Cut points & Assignment ARI \\\\",
                    "\\midrule",
                    *sensitivity_rows,
                    "\\bottomrule",
                    "\\end{tabular}",
                ]
            )
        )

    if not proxy.empty:
        rows = []
        for skill in SKILL_ORDER:
            g = proxy[proxy["skill_category"] == skill]
            rows.append(
                f"{nice(skill)} & {g['Bimanual coordination'].mean():.2f} & {g['Task efficiency'].mean():.2f} & "
                f"{g['Instrument motion control'].mean():.2f} & {g['Workspace excursion'].mean():.2f} & "
                f"{g['Coordination-control composite'].mean():.2f} \\\\"
            )
        (TAB / "osats_proxy.tex").write_text(
            "\n".join(
                [
                    "\\begin{tabular}{lrrrrr}",
                    "\\toprule",
                    "Procedure group & Bimanual $\\uparrow$ & Efficiency $\\uparrow$ & Motion control $\\uparrow$ & Workspace compactness & Coordination and control score $\\uparrow$ \\\\",
                    "\\midrule",
                    *rows,
                    "\\bottomrule",
                    "\\end{tabular}",
                ]
            )
        )

    seg_summary_path = OUT / "phase_segment_summary.csv"
    if seg_summary_path.exists():
        seg_summary = pd.read_csv(seg_summary_path)
        rows = []
        for phase in PHASE_ORDER:
            g = seg_summary[seg_summary["phase"] == phase]
            if g.empty:
                continue
            r = g.iloc[0]
            rows.append(
                f"{nice(phase)} & {int(r['n_segments'])} & {r['mean_duration_s']:.2f} & {r['median_duration_s']:.2f} \\\\"
            )
        (TAB / "phase_segment_summary.tex").write_text(
            "\n".join(
                [
                    "\\begin{tabular}{lrrr}",
                    "\\toprule",
                    "Phase & Segments & Mean duration (s) & Median duration (s) \\\\",
                    "\\midrule",
                    *rows,
                    "\\bottomrule",
                    "\\end{tabular}",
                ]
            )
        )

    if not phase_kinematics.empty:
        rows = []
        for phase in ["reach", "grasp", "transfer", "place", "nudge"]:
            g = phase_kinematics[phase_kinematics["phase"] == phase]
            if g.empty:
                continue
            r = g.iloc[0]
            rows.append(
                f"{nice(phase)} & {int(r['n_trials'])} & {r['duration_s']:.0f} & "
                f"{r['tool1_speed_mm_s']:.1f} & {r['tool2_speed_mm_s']:.1f} & "
                f"{r['combined_path_rate_mm_s']:.1f} \\\\"
            )
        (TAB / "phase_kinematics_summary.tex").write_text(
            "\n".join(
                [
                    "\\begin{tabular}{lrrrrr}",
                    "\\toprule",
                    "Phase & Trials & Duration (s) & Left tool speed & Right tool speed & Combined path rate \\\\",
                    "\\midrule",
                    *rows,
                    "\\bottomrule",
                    "\\end{tabular}",
                ]
            )
        )

    if not clusters.empty and not phase_trials.empty and not phase_cycles.empty:
        phase = _phase_trials_with_clusters(phase_trials, clusters).dropna(subset=["metric_cluster"])
        cyc = _phase_cycles_with_clusters(phase_cycles, clusters).dropna(subset=["metric_cluster"])
        if not phase.empty and not cyc.empty:
            trial_cycle = (
                cyc.groupby(["metric_cluster", "trial_key"], as_index=False)
                .agg(
                    mean_cycle_duration_s=("duration_s", "mean"),
                    mean_transitions=("n_transitions", "mean"),
                    n_cycles=("cycle_index", "count"),
                )
            )
            rows = []
            for band in BAND_ORDER:
                pg = phase[phase["metric_cluster"] == band]
                cg = cyc[cyc["metric_cluster"] == band]
                tg = trial_cycle[trial_cycle["metric_cluster"] == band]
                if pg.empty or cg.empty:
                    continue
                rows.append(
                    f"{BAND_LABELS[band]} & {pg['trial_key'].nunique()} & {len(cg)} & "
                    f"{tg['mean_cycle_duration_s'].mean():.2f} & {tg['mean_transitions'].mean():.2f} & "
                    f"{100 * pg['frac_transfer'].mean():.1f}\\% & {100 * pg['frac_place'].mean():.1f}\\% \\\\"
                )
            (TAB / "phase_bottlenecks.tex").write_text(
                "\n".join(
                    [
                        "\\begin{tabular}{lrrrrrr}",
                        "\\toprule",
                        "Motion-score band & Trials & Cycles & Cycle time (s) & Transitions & Transfer time & Place time \\\\",
                        "\\midrule",
                        *rows,
                        "\\bottomrule",
                        "\\end{tabular}",
                    ]
                )
            )

            drop_rows = []
            for band in BAND_ORDER:
                cg = cyc[cyc["metric_cluster"] == band]
                if cg.empty:
                    continue
                per_trial = cg.groupby("trial_key").agg(
                    n_drop_events=("n_drop_events", "sum"),
                    n_cycles=("cycle_index", "count"),
                )
                per_trial["rate"] = 100 * per_trial["n_drop_events"] / per_trial["n_cycles"]
                drop_rows.append(
                    f"Motion-score band & {BAND_SHORT_LABELS[band]} & "
                    f"{len(per_trial)} & {len(cg)} & {int(cg['n_drop_events'].sum())} & "
                    f"{per_trial['rate'].mean():.2f} & "
                    f"{100 * (per_trial['n_drop_events'] > 0).mean():.1f}\\% \\\\"
                )
            for skill in SKILL_ORDER:
                cg = cyc[cyc["skill_category"] == skill]
                if cg.empty:
                    continue
                per_trial = cg.groupby("trial_key").agg(
                    n_drop_events=("n_drop_events", "sum"),
                    n_cycles=("cycle_index", "count"),
                )
                per_trial["rate"] = 100 * per_trial["n_drop_events"] / per_trial["n_cycles"]
                drop_rows.append(
                    f"Procedure count & {nice(skill)} & {len(per_trial)} & {len(cg)} & "
                    f"{int(cg['n_drop_events'].sum())} & "
                    f"{per_trial['rate'].mean():.2f} & "
                    f"{100 * (per_trial['n_drop_events'] > 0).mean():.1f}\\% \\\\"
                )
            (TAB / "drop_episode_summary.tex").write_text(
                "\n".join(
                    [
                        "\\begin{tabular}{llrrrrr}",
                        "\\toprule",
                        "Grouping & Group & Trials & Cycles & Drop episodes & "
                        "Mean episodes per 100 cycles & Trials with drop \\\\",
                        "\\midrule",
                        *drop_rows,
                        "\\bottomrule",
                        "\\end{tabular}",
                    ]
                )
            )


def write_reproducibility_manifest() -> None:
    """Write an auditable manifest for the caches used by this manuscript."""
    input_paths = analysis_input_paths()
    packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "pyarrow",
    ]
    manifest = {
        "analysis_script": {
            "path": str((ROOT / "scripts" / "build_scirep_analysis.py").relative_to(ROOT)),
            "sha256": file_sha256(ROOT / "scripts" / "build_scirep_analysis.py"),
        },
        "python": {
            "executable": Path(sys.executable).name,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "packages": {name: package_version(name) for name in packages},
        "inputs": {
            key: {
                "path": portable_input_path(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "sha256": file_sha256(path),
            }
            for key, path in input_paths.items()
        },
        "random_seeds": {
            "bootstrap_and_sampling": 20260618,
            "classification_cv": 13,
            "pca": 20260618,
            "kmeans": 20260618,
        },
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    validate_analysis_inputs()
    setup_style()
    trials, cycles, frames = load_phase_data()
    all_df = load_all_trial_data(frames)
    feature_stats = feature_statistics(all_df)
    classification = run_classification(all_df)
    proxy = osats_proxy_scores(all_df)
    adjusted_models = cohort_adjusted_models(all_df, proxy)
    feature_dictionary = write_feature_dictionary(all_df)
    clusters, cluster_summary = kmeans_skill_clusters(all_df, proxy)
    jerk_checks = jerk_exclusion_sensitivity(
        all_df,
        proxy,
        clusters,
        cluster_summary,
    )
    trim_validation = trimming_rule_validation(frames, clusters)
    processing_checks = processing_sensitivity(
        all_df,
        frames,
        proxy,
        clusters,
        cluster_summary,
    )
    held_out_band_statistics(clusters)
    metric_band_classification = run_metric_band_classification(clusters)
    motion_sample = load_origin_motion(all_df)
    phase_statistics(trials, cycles, frames)
    phase_kinematics = phase_specific_kinematics(frames, clusters)
    plot_training_systems_context()
    plot_collection_setups()
    plot_task_setup_overview()
    plot_peg_transfer_cycle_placeholder()
    plot_cohort(all_df, trials)
    plot_feature_effects(feature_stats)
    plot_phase_timing(cycles, trials)
    plot_experience_links(all_df, trials)
    plot_osats_proxy(proxy, clusters, cluster_summary)
    plot_kmeans_clusters(clusters, cluster_summary)
    plot_motion_domains(all_df)
    plot_jaw_and_rotation(all_df)
    plot_workspace_heatmaps(motion_sample)
    plot_skill_workspace_xy(motion_sample)
    plot_phase_by_cluster(trials, clusters)
    plot_tool_specific_motion_boxplots(all_df)
    plot_sorted_participant_bars(all_df, proxy)
    plot_dataset_metric_matrix(all_df, proxy)
    plot_procedure_motion_disagreement(clusters)
    plot_phase_bottlenecks(trials, cycles, clusters)
    plot_cohort_effect_forest(all_df)
    plot_time_synchrony_quadrants(all_df, clusters)
    plot_continuous_performance_validation(proxy, clusters)
    write_tables(
        all_df,
        trials,
        cycles,
        feature_stats,
        adjusted_models,
        phase_kinematics,
        classification,
        proxy,
        clusters,
        cluster_summary,
        metric_band_classification,
    )
    write_reproducibility_manifest()
    summary = {
        "all_trials": int(len(all_df)),
        "phase_trials": int(len(trials)),
        "phase_cycles": int(len(cycles)),
        "feature_dictionary_rows": int(len(feature_dictionary)),
        "trimming_rule_validation": trim_validation.to_dict(orient="records"),
        "processing_sensitivity": processing_checks.to_dict(
            orient="records"
        ),
        "jerk_exclusion_sensitivity": jerk_checks,
        "classification": classification,
        "metric_band_classification": metric_band_classification,
        "kmeans": cluster_summary,
        "core_motion_score_mean_by_procedure_group": (
            proxy.groupby("skill_category")["Coordination-control composite"].mean().to_dict()
        ),
        "top_feature_effects": feature_stats.head(5).to_dict(orient="records"),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()
