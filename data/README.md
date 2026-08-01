# Analysis data

## Public contents

`derived/` contains aggregate tables and machine-readable summaries generated
from the complete collection. These files support review of the reported
statistics without exposing participant-level rows that are not yet available
in the staged Zenodo release.

Important files:

| File | Contents |
|:--|:--|
| `analysis_manifest.json` | Software versions, random seeds, input hashes, and portable input locations |
| `analysis_summary.json` | Main denominators, model summaries, clustering diagnostics, and sensitivity checks |
| `all_trial_feature_statistics.csv` | Effect sizes, confidence intervals, fixed- and random-effects correlation summaries, and corrected probabilities for 29 features |
| `cohort_specific_effects.csv` | Cohort-specific novice-versus-expert estimates |
| `cohort_adjusted_models.csv` | Cohort-adjusted procedure-volume models |
| `feature_dictionary.csv` | Feature labels, interpretation, direction, domain, and availability |
| `classification_summary.json` | Procedure-group classification checks |
| `metric_band_classification_summary.json` | Procedure-based targets and circular motion-band reconstruction checks |
| `kmeans_cluster_summary.csv` | Descriptive motion-score band sizes and centres |
| `kmeans_validation.csv` | Internal indices for alternative values of `k` |
| `processing_sensitivity.csv` | Position and orientation processing sensitivity |
| `jerk_exclusion_score_sensitivity.csv` | Continuous-score and band-assignment sensitivity after removing duration-dependent normalised jerk |
| `jerk_exclusion_contextual_correlations.csv` | Within-cohort duration and task-efficiency checks for the score without normalised jerk |
| `trimming_rule_validation.csv` | Automatic versus labelled task-boundary agreement |
| `phase_kinematics_summary.csv` | Aggregate phase-specific duration, speed, and path-rate summaries |

## Inputs required by the full script

The complete analysis entry point expects two roots.

### Phase cache

Set `LASK_PHASE_CACHE` to a directory containing:

| File | Minimum fields |
|:--|:--|
| `canonical_trials.csv` | `dataset`, `trial_name`, `trial_number`, `total_procedures`, `skill_category`, inclusion flags |
| `trials.parquet` | Recording identifiers, frame count, duration, cycle count, phase fractions, procedure metadata |
| `cycles.parquet` | Recording identifier, cycle index, duration, transitions, and phase fractions |
| `frames.parquet` | Recording identifier, frame index, phase labels, cycle index, and tool-specific labels |

### AI-ELT root

Set `LASK_AI_ELT_ROOT` to the AI-ELT project root containing:

```text
outputs/
├── motion/
│   └── combined_motion_metrics.csv
├── papers/paper1/results/
│   └── extended_features_cache_trimmed.csv
└── ssl/origin/origin_data/ORIGIN_ALL/
    └── <recording>.json
```

The per-recording JSON files contain a dense `features` matrix. Its first
14 columns are:

```text
tool1 x, y, z, qw, qx, qy, qz,
tool2 x, y, z, qw, qx, qy, qz
```

`LASK_ORIGIN_MOTION` can override that JSON directory.

## Output policy

The full script also writes participant-level intermediates. These remain
ignored by Git until the corresponding recordings and metadata are released
through the governed Zenodo record. Aggregate tables are tracked.

When the Zenodo release is expanded, the release process should include:

1. an analysis-ready manifest mapping public trial names to analysis keys;
2. the complete phase cache or a deterministic builder;
3. a converter from released kinematic CSV files to the per-recording analysis
   input;
4. checksums matching `analysis_manifest.json`;
5. a rerun of the complete analysis from a clean environment.
