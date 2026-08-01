# Motion feature definitions

## Overview

The analysis converts dense three-dimensional position and orientation
measurements into interpretable summaries of instrument use. This page defines
the main quantities in plain language and gives the calculation where it
matters.

The machine-readable dictionary is
[`data/derived/feature_dictionary.csv`](../data/derived/feature_dictionary.csv).
Its display directions are analytical hypotheses, not validated clinical
preferences.

The prose and figures use **left tool** and **right tool** for the channels that
normally enter from the left and right sides of the endoscopic image. These
labels do not indicate hand dominance. Internal feature names retain
`tool1_*` and `tool2_*` for compatibility.

## Preprocessing

For Paediatric and Urology 2, time is calculated at 13 frames per second.
Urology 1 uses 26 frames per second.

Position is smoothed separately on each axis using a third-order
Savitzky-Golay filter over approximately 0.4 seconds. This corresponds to
5 frames at 13 frames per second and 11 frames at 26 frames per second. The
filter fits a short polynomial within each moving window, reducing
frame-to-frame noise while retaining the overall path.

The analysed task interval starts 0.5 seconds before the first period of
sustained motion above 2 mm/s and ends 0.5 seconds after the final sustained
period. The sustained interval must last approximately 0.4 seconds. The same
automatic rule is applied to all primary recordings.

Dense phase labels are used to validate, but not define, the primary automatic
interval. In 34 phase-labelled primary recordings, median interval overlap was
0.968 and the median absolute duration difference was 4.65 seconds.

## Single-instrument features

Let `p(t)` be tool-tip position at time `t`.

### Speed

Velocity is the rate of change of position:

```text
v(t) = dp(t) / dt
speed(t) = ||v(t)||
```

Average speed is the mean speed across the analysed interval.

### Path length

Path length is the sum of the three-dimensional distances between consecutive
positions:

```text
path length = sum ||p(i) - p(i-1)||
```

It measures total instrument travel in millimetres. A shorter path is not
automatically better because the required path depends on object and peg
locations.

### Acceleration and jerk

Acceleration is the rate of change of velocity. Jerk is the rate of change of
acceleration. Abrupt starts, stops, or corrections increase jerk.

The dimensionless normalised jerk used here is:

```text
normalised jerk =
sqrt(0.5 * integral(||jerk(t)||^2 dt) * duration^5 / path_length^2)
```

Duration and path length appear in the normalisation. Normalised jerk is
therefore not independent of task time.

### Active and idle fractions

A tool is considered active at a frame when speed is at least 5 mm/s. Its
active fraction is the proportion of active frames. Idle episodes are
contiguous periods below 5 mm/s lasting at least 0.25 seconds.

### Speed peaks

Speed peaks are local maxima above 5 mm/s with at least 2 mm/s prominence and
at least 0.25 seconds separation. They provide a simple description of
stop-start movement, not a direct count of corrective actions.

### Workspace

- Image-plane working area is the `x` range multiplied by the `y` range.
- Working volume is the product of the `x`, `y`, and `z` ranges.
- Depth range is the range along the camera axis.

These are axis-aligned bounding measures. They do not reconstruct the exact
shape or occupied volume of the path.

### Rotation

The change in orientation between two frames is obtained from the relative
rotation between their unit quaternions. Total rotation is the sum of the
frame-to-frame rotation magnitudes in degrees.

Rotation per path is:

```text
total rotation in degrees / path length in millimetres
```

Angular velocity variability is the standard deviation of non-zero angular
speed divided by its mean.

## Bimanual features

### Bimanual correlation

Bimanual correlation is the Pearson correlation between the simultaneous left-
and right-tool speed traces. A positive value means the tools tend to speed up
and slow down together. It does not show that the instruments follow the same
spatial path.

### Bimanual lag

Cross-correlation compares one speed trace with shifted copies of the other.
The shift with the greatest similarity, searched within plus or minus
5 seconds, defines the bimanual lag. A lag of zero means the strongest
similarity occurs without shifting either trace. A larger lag indicates a
greater timing offset, but it does not identify which hand caused it.

### Simultaneous motion

Frames above 10 mm/s are treated as clearly moving:

```text
simultaneous motion =
frames where both tools move / frames where either tool moves
```

### Concurrent efficiency

When both tools move faster than 5 mm/s, their speed balance is calculated as:

```text
minimum(tool speeds) / maximum(tool speeds)
```

The reported value is the mean across those frames. A value near one indicates
similar concurrent speeds. This is called concurrent efficiency rather than
concurrent tool use because it measures speed balance during concurrent
movement.

### Bimanual symmetry

Bimanual symmetry is the shorter tool path divided by the longer tool path.
It approaches one when path lengths are similar. Equal path lengths are not
necessarily optimal for every transfer.

### Tool-distance variability

The distance between tool tips is calculated at each frame. Its coefficient of
variation is the standard deviation divided by the mean. This describes the
stability of inter-tool spacing.

## Motion domains

The paper uses four descriptive domains. Closely related or mirrored tool
measurements are averaged into constructs first so that duplicate channels do
not receive extra weight.

### Bimanual coordination

Equal-weighted constructs:

1. bimanual speed correlation;
2. simultaneous motion and concurrent efficiency;
3. stability of the distance between tools.

### Instrument motion control

Equal-weighted constructs:

1. normalised jerk for both tools;
2. rotation per path for both tools;
3. angular velocity variability for both tools.

Because normalised jerk includes duration in its definition, it is not an
independent measure of movement quality. Removing it produced a continuous
score that remained correlated with the primary score (`rho = 0.853`) but
changed many categorical assignments (adjusted Rand index `0.351`). None of
the six alternative-score associations with duration or task efficiency
survived correction. The continuous constructs should therefore be reported
directly, and the descriptive bands should not be treated as fixed grades.

### Task efficiency

Equal-weighted constructs:

1. analysed duration;
2. idle and active fractions;
3. speed-peak counts.

### Descriptive workspace compactness

Equal-weighted constructs:

1. working volumes;
2. image-plane working areas;
3. depth ranges.

No clinical preference for a smaller workspace is assumed because appropriate
excursion depends on task phase and trainer geometry.

## Coordination and control score

Each feature is first oriented according to its prespecified display direction
and converted to a percentile within its acquisition cohort. Let `r` be the
mean construct percentile. A domain is displayed on a relative 1 to 5 scale:

```text
domain score = 1 + 4 * r
```

The coordination and control score is the equal-weighted mean of bimanual
coordination and instrument motion control. Task efficiency and workspace are
not entered as separate score features.

This score is relative to the observed cohort. It is not a validated Objective
Structured Assessment of Technical Skills, Global Operative Assessment of
Laparoscopic Skills, Global Evaluative Assessment of Robotic Skills, McGill
Inanimate System for Training and Evaluation of Laparoscopic Skills, or
Fundamentals of Laparoscopic Surgery rating.

## Motion-score bands

K-means groups similar score values around a chosen number of centres. With
three centres, it partitions the continuous coordination and control score
into descriptive bands:

| Band | Recordings | Centre | Observed cut point |
|:--|--:|--:|--:|
| Lower | 16 | 2.16 | Below 2.51 |
| Middle | 41 | 2.87 | 2.51 to 3.18 |
| Upper | 50 | 3.50 | Above 3.18 |

The cut points are conditional on the current score construction and cohort
percentiles. They should be recalculated when the dataset expands. Two- and
four-band solutions also had plausible internal indices, so three bands are
retained for interpretability rather than claimed as natural clinical classes.
