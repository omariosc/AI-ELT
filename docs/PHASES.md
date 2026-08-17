# Phase annotation protocol

## Purpose

Dense phase labels divide a peg-transfer recording into observable actions.
They support cycle timing, action composition, and phase-specific kinematic
analysis. They are not ratings of how well an action was performed.

## Annotation process

One researcher, O.C., annotated the phases. The definitions were developed
through direct instruction and discussion with surgeons at the urology and
paediatric boot camps, together with observation of peg-transfer performance
at those events.

Participant seniority and procedure-count metadata were not visible during
annotation. Differences in apparent confidence and fluency could still be
perceived from the videos, and the annotator sometimes formed informal
impressions of skill. Those impressions were not recorded and were not used to
define the phase labels.

No second annotator or adjudication process has yet been completed. Inter-rater
agreement therefore cannot currently be reported.

## Labels

| Phase | Operational definition |
|:--|:--|
| Reach | Movement towards an object before attempting control |
| Grasp | Attempt to obtain control of an object |
| Transfer | Movement of an object between instruments |
| Place | Positioning or releasing an object at a destination peg |
| Nudge | Corrective contact intended to reposition an object |
| Idle | No task-directed instrument action |
| Dropped | Visible loss of object control |

Per-frame exports can also contain tool-specific labels, cycle index, active
tool, and events. The `active_tool` field in an earlier consolidated export is
zero for every frame and is not used by the paper analysis. Tool activity is
instead derived from speed or from the phase label, depending on the analysis.

## From frames to cycles

A candidate cycle is retained only when:

1. it contains a `place` label; and
2. it contains at least one `reach`, `grasp`, `transfer`, or `nudge` label.

An exporter could occasionally place a boundary segment containing only
placement frames into the next cycle index. Such a segment is attached to the
immediately preceding interval only when that interval has no placement.
Other placement-only residuals and attempts without placement are excluded.

This rule produced:

| Analysis set | Recordings | Placement-complete cycles |
|:--|--:|--:|
| Full phase inventory | 38 | 425 |
| Primary phase set | 34 | 383 |

Cycle measurements are first averaged within recording and then across
recordings. A participant with more completed transfers therefore does not
receive more weight in recording-level comparisons.

## Current interpretation

Phase results are descriptive. The annotated subset is incomplete and its
motion-score composition is imbalanced. In particular, only four recordings in
the lower motion-score band enter the primary phase comparison. Expanded
annotation and an independent reliability sample are planned.

Phase labels may be used to locate where time, corrections, or dropped-object
episodes occur. They should not be converted directly into clinical competence
grades without expert validation.

Some individual recordings carry defects that restrict how their labels may be
used, for example a recording that may be trained on but must be kept out of
validation. These are documented in
[annotation caveats and recording usage notes](ANNOTATION_CAVEATS.md).
