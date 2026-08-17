# Annotation caveats and recording usage notes

This document records recording-level defects in the phase and instrument
annotations that constrain how a recording may be used. It is the companion to
the [phase annotation protocol](PHASES.md), which describes how the labels were
produced. The caveats here describe where individual recordings should be
restricted, most importantly when a recording may be used for training but must
be kept out of validation, test, or reliability sets.

The list will grow as further recordings and phase labels are added in later
dataset versions. Each release that adds annotations should also add any
recording-level caveats here.

## Usage roles

| Role | Meaning |
|:--|:--|
| Training only | The recording may appear in a training split. It must be excluded from any validation, test, or inter-rater reliability set, because a defect makes its labels unreliable as ground truth for evaluation. |
| Exclude | The recording, or the affected frames, should not be used at all. |

These roles correspond to the per-frame annotation flags used in the internal
pipeline: `exclude` marks frames that may be trained on but never validated, and
`broken` marks frames that are dropped everywhere. Where only part of a
recording is affected, the affected frames carry the flag and the rest of the
recording is unrestricted.

## Recording caveats

| Cohort | Internal identifier | Release identifier | Affected span | Issue | Recommended role |
|:--|:--|:--|:--|:--|:--|
| Paediatric | `BAPES2024/Industry/Trial18` | To be assigned; not in version 1.0 | Start of the first transfer cycle | Frames were skipped at the beginning of the recording, so the opening phase boundaries of the first cycle are incomplete and cannot be trusted as ground truth. | Training only |

## Notes

The Paediatric caveat above is the reason recording `Trial18` is absent from the
current phase-labelled analysis set. When it is released with phase labels it
should enter the training split only, and its skipped opening frames should be
marked so that any split builder keeps them out of validation and test.
