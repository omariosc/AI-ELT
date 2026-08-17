# LASK — repo working notes

Public **release + reproducible analysis** for the LASK laparoscopic peg-transfer
skill dataset (video + Aurora electromagnetic instrument kinematics + dense phase
labels). Concept DOI `10.5281/zenodo.20752650`; public v1.0 = 37 recordings, with
further recordings and phase labels staged into later Zenodo versions. This repo
holds docs, the analysis scripts, and the tracked **aggregate** result tables —
never participant-level rows (those ship only through the governed Zenodo record).

## Cohort naming (easy to trip on)

The public release renames the internal capture sets. They are the same data:

| Release cohort | Internal name | Split role | Rate |
|---|---|---|---|
| Paediatric | `BAPES2024` | validation | 13 fps |
| Urology 1 | `6DOF2023` | testing | ~26 fps, no jaw channel |
| Urology 2 | `7DOF2024` | training | 13 fps |

Full collection = 115 recordings (107 in the primary motion set); phase-labelled
subset = 38 (primary phase set = 34). Six-DoF = position + orientation; the 7th
channel is relative jaw opening.

## Docs map

- [`docs/DATASET.md`](docs/DATASET.md) — cohorts, archive layout, kinematic/annotation files.
- [`docs/PHASES.md`](docs/PHASES.md) — phase-label protocol (reach/grasp/transfer/place/nudge/idle/dropped).
- [`docs/FEATURES.md`](docs/FEATURES.md) — motion-feature definitions.
- [`docs/ANALYSIS.md`](docs/ANALYSIS.md) — the quantitative-analysis writeup.
- [`docs/ANNOTATION_CAVEATS.md`](docs/ANNOTATION_CAVEATS.md) — recording-level defects that restrict use (see below).

## Annotation caveats / train-only recordings

Recording-level defects that constrain how a recording may be used are recorded in
`docs/ANNOTATION_CAVEATS.md` (cross-linked from DATASET.md limitations and PHASES.md).
A **"training only"** recording may appear in a training split but must be kept out
of any validation, test, or inter-rater reliability set, because a defect makes its
labels unreliable as evaluation ground truth. First entry: Paediatric `BAPES2024/
Industry/Trial18` — frames were skipped at the start of its first cycle, so the
opening phase boundaries are unusable; it is absent from the current phase-labelled
set and, when released, must enter the training split only. Add future train-only /
exclude recordings to that file as annotations are released.

## Analysis scripts

Run from the repo root. Both expect two roots via env vars (see `data/README.md`):
`LASK_PHASE_CACHE` (canonical_trials.csv + trials/cycles/frames parquets) and
`LASK_AI_ELT_ROOT` (motion metrics + per-recording feature JSON); optional
`LASK_ORIGIN_MOTION`.

- `scripts/build_scirep_analysis.py` — the complete analysis entry point; regenerates
  the tracked aggregate tables in `data/derived/` and figures in `paper/figures/`.
- `scripts/analyze_phase_motion_features.py` — phase-specific motion at the
  **recording** level (features computed within cycle×phase, then averaged to one row
  per recording; recordings, not frames/cycles, are the unit of analysis). Transfer
  features are exported but excluded from cross-cohort modelling because Urology 1's
  protocol folded transfer into placement for some recordings. Produces the
  consistent experience-ordered patterns, the phase-specific novice/expert classifier,
  and the jaw-voltage analysis (`consistent_experience_patterns*.csv`,
  `phase_motion_*.csv`, `jaw_*.csv`, `fig3`, `fig8`). Fixed `RANDOM_SEED`; matplotlib
  is headless (`MPLCONFIGDIR` under `paper/build/`).
- `scripts/validate_release.py` — release/consistency validator.

## data/derived and paper/figures are gitignore-whitelisted

`.gitignore` ignores `data/derived/*` and `paper/figures/*` wholesale, then re-includes
specific files with `!` lines. Only **aggregate** result tables and release figures are
tracked; participant-level intermediates stay untracked until the Zenodo release. When a
script emits a new aggregate table or figure that should ship, add a matching `!` line —
otherwise it stays invisible to git.
