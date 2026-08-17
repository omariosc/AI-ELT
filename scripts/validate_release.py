#!/usr/bin/env python3
"""Validate the aggregate LASK analysis release without private source data."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"


def read_json(name: str) -> dict:
    with (DERIVED / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(name: str) -> list[dict[str, str]]:
    with (DERIVED / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    required_files = [
        ROOT / "README.md",
        ROOT / "CITATION.cff",
        ROOT / "docs" / "DATASET.md",
        ROOT / "docs" / "ANALYSIS.md",
        ROOT / "docs" / "FEATURES.md",
        ROOT / "docs" / "PHASES.md",
        ROOT / "scripts" / "build_scirep_analysis.py",
        ROOT / "scripts" / "analyze_phase_motion_features.py",
        DERIVED / "analysis_manifest.json",
        DERIVED / "analysis_summary.json",
        DERIVED / "feature_dictionary.csv",
        DERIVED / "all_trial_feature_statistics.csv",
        DERIVED / "cohort_adjusted_models.csv",
        DERIVED / "consistent_experience_patterns.csv",
        DERIVED / "consistent_experience_pattern_medians.csv",
        DERIVED / "jerk_exclusion_contextual_correlations.csv",
        DERIVED / "jerk_exclusion_score_sensitivity.csv",
        DERIVED / "phase_motion_associations.csv",
        DERIVED / "phase_motion_nested_summary.csv",
    ]
    for path in required_files:
        require(path.is_file() and path.stat().st_size > 0, f"Missing required file: {path}")

    summary = read_json("analysis_summary.json")
    require(summary["all_trials"] == 115, "Expected 115 recordings in the inventory")
    require(summary["phase_trials"] == 38, "Expected 38 phase-labelled recordings")
    require(summary["phase_cycles"] == 425, "Expected 425 placement-complete cycles")
    require(summary["feature_dictionary_rows"] == 30, "Expected 30 defined motion features")

    feature_rows = read_csv("all_trial_feature_statistics.csv")
    require(len(feature_rows) == 29, "Expected 29 inferential feature rows")
    significant_correlations = [
        row for row in feature_rows if float(row["spearman_q"]) < 0.05
    ]
    significant_group_tests = [
        row for row in feature_rows if float(row["kruskal_q"]) < 0.05
    ]
    require(
        len(significant_correlations) == 8,
        "Expected eight false-discovery-rate-corrected procedure correlations",
    )
    require(
        not significant_group_tests,
        "No three-group feature test should survive correction",
    )

    ordered_rows = read_csv("consistent_experience_patterns.csv")
    require(len(ordered_rows) == 4, "Expected four convergent experience patterns")
    require(
        {row["feature"] for row in ordered_rows}
        == {
            "total_time",
            "tool1_normalized_jerk",
            "tool2_normalized_jerk",
            "tool2_num_speed_peaks",
        },
        "Unexpected convergent experience patterns",
    )
    require(
        all(float(row["spearman_q"]) < 0.05 for row in ordered_rows),
        "Each convergent pattern must retain a corrected continuous association",
    )

    bimanual = next(
        row for row in feature_rows if row["feature"] == "bimanual_correlation"
    )
    require(
        float(bimanual["spearman_random_ci_low"]) < 0
        < float(bimanual["spearman_random_ci_high"]),
        "Bimanual random-effects interval should include no association",
    )

    jerk_summary = read_csv("jerk_exclusion_score_sensitivity.csv")
    require(len(jerk_summary) == 1, "Expected one jerk-exclusion summary row")
    jerk_row = jerk_summary[0]
    require(int(jerk_row["n"]) == 107, "Expected 107 recordings in jerk sensitivity")
    require(
        0.84 < float(jerk_row["score_spearman_rho"]) < 0.87,
        "Unexpected jerk-exclusion score correlation",
    )
    require(
        0.34 < float(jerk_row["fixed_cut_assignment_ari"]) < 0.37,
        "Unexpected jerk-exclusion band agreement",
    )
    jerk_context = read_csv("jerk_exclusion_contextual_correlations.csv")
    require(len(jerk_context) == 6, "Expected six jerk-exclusion contextual checks")
    require(
        min(float(row["q"]) for row in jerk_context) >= 0.05,
        "No jerk-exclusion contextual check should survive correction",
    )

    model_rows = read_csv("cohort_adjusted_models.csv")
    significant_models = {
        row["outcome"] for row in model_rows if float(row["procedure_q"]) < 0.05
    }
    require(
        significant_models
        == {
            "total_time",
            "bimanual_correlation",
            "Coordination-control composite",
        },
        "Unexpected set of corrected cohort-adjusted procedure effects",
    )

    jaw_rows = read_csv("jaw_voltage_statistics.csv")
    require(len(jaw_rows) == 7, "Expected seven whole-recording jaw checks")
    require(
        all(int(row["n"]) == 83 for row in jaw_rows),
        "Expected 83 primary recordings in each jaw check",
    )
    require(
        min(float(row["q"]) for row in jaw_rows) >= 0.05,
        "No whole-recording jaw check should survive correction",
    )
    jaw_sensitivity = read_csv("jaw_smoothing_sensitivity.csv")
    require(
        len(jaw_sensitivity) == 21,
        "Expected seven jaw checks under three smoothing windows",
    )
    require(
        {int(float(row["window_frames"])) for row in jaw_sensitivity} == {5, 7, 11},
        "Unexpected jaw smoothing windows",
    )
    jaw_phases = read_csv("jaw_phase_comparisons.csv")
    require(len(jaw_phases) == 4, "Expected one overall and three planned jaw-phase tests")

    phase_associations = read_csv("phase_motion_associations.csv")
    require(len(phase_associations) == 10, "Expected ten prespecified phase associations")
    corrected_phase = [row for row in phase_associations if float(row["q_value"]) < 0.05]
    require(
        len(corrected_phase) == 1
        and corrected_phase[0]["feature"] == "transport_place_speed_peaks_per_s",
        "Unexpected corrected phase-motion association",
    )
    phase_model = read_csv("phase_motion_nested_summary.csv")
    require(len(phase_model) == 1, "Expected one nested phase-model summary")
    require(
        0.88 < float(phase_model[0]["macro_balanced_accuracy"]) < 0.90,
        "Unexpected phase-model cohort-mean balanced accuracy",
    )
    require(
        0.82 < float(phase_model[0]["pooled_balanced_accuracy"]) < 0.84,
        "Unexpected phase-model pooled balanced accuracy",
    )

    manifest = read_json("analysis_manifest.json")
    analysis_script = ROOT / manifest["analysis_script"]["path"]
    require(
        sha256(analysis_script) == manifest["analysis_script"]["sha256"],
        "Analysis script hash does not match the public manifest",
    )
    for source in manifest["inputs"].values():
        require(
            not source["path"].startswith("/"),
            "Manifest contains a machine-specific absolute input path",
        )
        require(len(source["sha256"]) == 64, "Input hash is missing or malformed")

    figures = [
        "fig1_cohort_structure.png",
        "fig2_feature_effects.png",
        "fig3_consistent_experience_patterns.png",
        "fig3_phase_timing.png",
        "fig4_experience_links.png",
        "fig8_jaw_voltage_analysis.png",
        "fig21_cohort_effect_forest.png",
        "fig23_continuous_performance_validation.png",
    ]
    for name in figures:
        path = ROOT / "paper" / "figures" / name
        require(path.is_file() and path.stat().st_size > 20_000, f"Missing figure: {name}")

    print(
        "Validated LASK aggregate release: "
        "115 recordings, 38 phase recordings, 425 cycles, "
        "29 feature tests, 8 corrected continuous associations, "
        "4 convergent ordered patterns, 10 phase tests, "
        "phase-model balanced accuracy 0.89."
    )


if __name__ == "__main__":
    main()
