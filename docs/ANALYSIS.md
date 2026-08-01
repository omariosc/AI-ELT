# Quantitative peg-transfer analysis

## Research question

The analysis asks which measurable aspects of instrument motion are associated
with laparoscopic experience during peg transfer, and whether they can support
transparent quantitative feedback.

The intended clinical use is not to replace an educator with an unvalidated
score. It is to identify reproducible measurements that could complement
expert observation, make feedback more specific, and support later automated
assessment after external validation.

The analysis calls the acquisition channels the **left tool** and **right
tool**, based on their usual side of entry in the endoscopic image. These are
not dominant-hand labels. The mapping was checked against annotated reference
frames from every cohort. Machine-readable tables retain `tool1_*` and
`tool2_*` field names for compatibility with the released files.

## Analysis populations

Four populations are kept separate:

| Population | Recordings | Study identifiers | Cycles | Use |
|:--|--:|--:|--:|:--|
| Complete inventory | 115 | 111 | Not applicable | Dataset description |
| Primary motion set | 107 | 107 | Not applicable | Motion inference and modelling |
| Phase inventory | 38 | 37 | 425 | All available dense phase labels |
| Primary phase set | 34 | 34 | 383 | Recording-level phase comparisons |

The study identifiers cannot prove that a participant who attended two
different events received the same identifier. The paper therefore refers to
recordings rather than claiming 107 confirmed unique people.

## Cohort handling

The collection settings differ in tools, cameras, trainers, frame rate, and
coordinate distributions. Paediatric, Urology 1, and Urology 2 are not treated
as interchangeable samples from one acquisition system.

The analysis therefore uses:

- descriptive plots separated by cohort;
- feature correlations calculated within cohort and combined afterwards;
- cohort-adjusted linear models;
- within-cohort percentiles when constructing relative motion domains;
- leave-one-cohort-out checks for exploratory models.

This approach retains common directional evidence without interpreting raw
millimetre, speed, or workspace values as directly comparable across rigs.

## Task interval and motion extraction

Blank start and end frames are excluded using one automatic sustained-motion
rule across all primary recordings. The interval begins shortly before the
first sustained instrument motion and ends shortly after the last. Dense phase
labels are used only to check this boundary rule.

Position is smoothed with an approximately 0.4-second Savitzky-Golay window.
Velocity, acceleration, jerk, path length, workspace range, rotation, active
time, and bimanual measurements are then calculated from the retained frames.
The complete feature definitions are in [`FEATURES.md`](FEATURES.md).

## Statistical analysis

### Feature screen

One constant feature was removed before inference, leaving 29 motion features.

For each feature:

1. novice, intermediate, and expert procedure groups were compared separately
   within each cohort using the Kruskal-Wallis rank test, which does not assume
   a normal distribution;
2. the three cohort probabilities were combined using Fisher's method, which
   tests whether the cohort results collectively depart from the null
   hypothesis;
3. associations with lifetime procedure count were estimated using Spearman
   rank correlation within each cohort;
4. correlations were combined using fixed-effect meta-analysis, which weights
   more precise cohort estimates more strongly;
5. false discovery rate correction was applied separately across the 29 group
   tests and 29 continuous correlations.

The corrected probability is reported as `q`. Cliff's delta compares novice
and expert procedure groups without assuming a normal distribution. It
estimates how often a value from one group exceeds a value from the other after
accounting for ties. Confidence intervals use 3,000 bootstrap resamples within
cohort.

Bimanual correlation had substantial between-cohort heterogeneity. A
random-effects sensitivity analysis therefore used DerSimonian-Laird
between-cohort variance and a modified Knapp-Hartung interval on Fisher's
`z` scale. Only three cohorts contribute, so this sensitivity interval is
necessarily imprecise.

### Cohort-adjusted models

Eight outcomes were selected to represent task duration, bimanual
coordination, speed, idle time, depth excursion, and the continuous
coordination and control score.

Each standardised outcome was regressed on:

- `log10(lifetime procedures + 1)`; and
- indicators for acquisition cohort.

A one-unit change in the procedure term represents an approximately tenfold
increase in procedures plus one. Confidence intervals use 2,000 bootstrap
resamples within cohort. Probabilities use 5,000 permutations of procedure
volume within cohort and are corrected across the eight models.

### Exploratory classification

Logistic regression and random forest models compare:

- duration alone;
- all 30 retained classifier inputs;
- fixed procedure-count groups;
- procedure tertiles;
- procedure clusters;
- data-derived motion-score bands.

Five-fold evaluation preserves approximate class proportions. A separate
leave-one-cohort-out evaluation trains on two cohorts and tests on the third.
Balanced accuracy is the mean sensitivity across classes, so each class
contributes equally despite unequal sample sizes.

## Main results

### Procedure groups did not define clear motion classes

No novice, intermediate, and expert feature comparison remained significant
after false discovery rate correction. This is an important result. It means
that the analysis cannot assume every high-volume participant performs the
recorded task better than every low-volume participant.

Procedure-group classification was also weak:

| Input and model | Five-fold balanced accuracy |
|:--|--:|
| Duration only, logistic regression | 0.45 +/- 0.17 |
| Duration only, random forest | 0.40 +/- 0.05 |
| All motion features, logistic regression | 0.36 +/- 0.07 |
| All motion features, random forest | 0.34 +/- 0.11 |

Adding more kinematic features did not rescue the noisy procedure-count target.
This guards against presenting case volume as ground truth skill.

### Continuous procedure volume retained modest associations

Eight of 29 within-cohort meta-analytic correlations remained significant:

| Feature | Spearman correlation | 95% confidence interval | Corrected `q` |
|:--|--:|:--|--:|
| Left-tool normalised jerk | -0.338 | -0.500 to -0.153 | 0.011 |
| Analysed duration | -0.327 | -0.491 to -0.141 | 0.011 |
| Right-tool normalised jerk | -0.301 | -0.469 to -0.112 | 0.020 |
| Right-tool speed peaks | -0.288 | -0.458 to -0.098 | 0.023 |
| Left-tool speed peaks | -0.282 | -0.453 to -0.092 | 0.023 |
| Left-tool path length | -0.278 | -0.449 to -0.087 | 0.023 |
| Right-tool speed | 0.265 | 0.074 to 0.438 | 0.026 |
| Bimanual correlation | 0.265 | 0.074 to 0.438 | 0.026 |

The effects are modest and distributions overlap. Bimanual correlation also
showed moderate between-cohort heterogeneity (`I2 = 60%`). Cohort-specific
correlations were -0.08 in Urology 1, 0.15 in Paediatric, and 0.44 in Urology
2. The random-effects sensitivity estimate was 0.21 with a 95% confidence
interval from -0.46 to 0.73. The `I2` statistic estimates the percentage of
observed variation attributable to differences between cohorts rather than
sampling uncertainty. This result should not be treated as a consistent
cross-cohort effect or universal threshold.

### Cohort-adjusted estimates

Three prespecified outcomes remained significant after correction:

| Outcome | Standardised change per tenfold procedure increase | 95% confidence interval | Corrected `q` |
|:--|--:|:--|--:|
| Analysed duration | -0.43 | -0.72 to -0.19 | 0.002 |
| Bimanual correlation | 0.29 | 0.09 to 0.51 | 0.030 |
| Coordination and control score | 0.28 | 0.07 to 0.51 | 0.033 |

These estimates identify duration and temporal coordination for further
validation after accounting for average cohort differences. They do not
establish causal effects of training, and the bimanual association remains
heterogeneous between cohorts.

### Acquisition setting can dominate raw movement

Cohort alone explained 88% of the variance in left-tool speed and 92% in
right-tool speed. Procedure volume added little after cohort for these
outcomes.

This result supports separate cohort plots and warns against a single raw-speed
threshold across equipment configurations.

## Data-derived motion-score bands

The coordination and control score combines bimanual coordination and
instrument motion control. Within-cohort percentiles reduce the effect of
acquisition scale. K-means with three centres gives:

| Band | Recordings | Centre | Procedure-count median |
|:--|--:|--:|--:|
| Lower | 16 | 2.16 | 40 |
| Middle | 41 | 2.87 | 20 |
| Upper | 50 | 3.50 | 80 |

Observed cut points are 2.51 and 3.18. Agreement with fixed procedure-count
groups is negligible (adjusted Rand index 0.019). The adjusted Rand index
measures agreement between two group assignments after allowing for chance. A
value of one indicates identical assignments, while a value near zero
indicates no greater agreement than expected by chance.

This mismatch is clinically informative. Defining skill from procedure count
alone can label a strong low-volume performance as novice and an inefficient
high-volume performance as expert. Motion and experience should therefore be
reported as related but distinct evidence.

The three-band solution has silhouette 0.534 and Davies-Bouldin index 0.557.
The silhouette score is higher when recordings are closer to their assigned
group than neighbouring groups. The Davies-Bouldin index is lower when groups
are compact and well separated. Two- and four-band solutions also have
plausible internal indices. Three bands are retained to aid interpretation,
not because the data prove three natural competence classes.

### Duration-dependent smoothness sensitivity

Normalised jerk contains analysed duration in its formula. The score was
therefore recalculated after removing the normalised-jerk construct from
instrument motion control.

| Sensitivity result | Value |
|:--|--:|
| Correlation with primary continuous score | 0.853 |
| Median absolute score change | 0.167 |
| Agreement using primary cut points | 0.351 |
| Agreement after refitting cut points | 0.351 |

The alternative score contained bimanual coordination, rotation per path, and
angular velocity variability, but no direct duration term. None of its six
within-cohort associations with analysed duration or task efficiency survived
false discovery rate correction (`q >= 0.096`). The continuous measurements
remain usable, but the categorical bands depend materially on normalised jerk
and should not be treated as calibrated grades.

Logistic regression and random forest recover the band rule with balanced
accuracy around 0.95. This is expected because the target bands are calculated
from the same motion domains supplied to the classifier. It confirms software
consistency but is not independent prediction and is not evidence of clinical
validity.

## Phase analysis

The phase subset describes where time and corrections occur within a transfer
cycle. Labels include reach, grasp, transfer, place, nudge, idle, and dropped
object.

Only four lower-band recordings enter the primary phase comparison. Drop
episodes are sparse, and several phase measures are non-monotonic across bands.
The phase results are therefore exploratory. Expanded annotation will improve
coverage, while an independently annotated sample is needed to quantify label
reliability.

The current phase analysis should be used to:

- identify actions responsible for long cycles;
- examine phase-specific speed or path rate;
- count visible drop episodes;
- develop candidate feedback for corrective nudges or repeated grasp attempts.

It should not yet be used to assign clinical grades.

## Potential value to surgeons and trainers

The current evidence prioritises three different roles for future validation.

- **Candidate feedback measurements:** duration, normalised jerk, stop-start
  speed peaks, travel distance, and bimanual correlation. Prospective studies
  must show whether presenting these values improves learning.
- **Contextual diagnostics:** phase timing, raw speed, rotation, workspace use,
  and jaw voltage. These may explain a performance pattern but are too
  acquisition-dependent or weakly calibrated to grade performance alone.
- **Unsupported as stand-alone skill labels:** procedure count, a single speed
  threshold, a universally small workspace, or reconstruction of the
  internally defined motion bands by a classifier.

A candidate report could show data quality first, then duration and visible
errors, left- and right-tool smoothness, stop-start peaks, travel distance,
bimanual coordination, and the phase responsible for delay or correction.
Repeated attempts could show whether each measure changes within the same
training setup. This report remains a design for prospective evaluation rather
than a validated intervention.

With further validation, the framework could support:

1. objective evidence alongside expert observation;
2. feedback that distinguishes timing, bimanual coordination, motion
   smoothness, and workspace use;
3. longitudinal tracking within the same simulator setup;
4. identification of a specific phase responsible for inefficient completion;
5. transparent comparison of training curricula without assuming procedure
   volume equals competence;
6. development of video-based systems trained against high-fidelity motion
   measurements.

The strongest near-term application is evaluation of formative feedback.
Summative assessment would require blinded expert ratings, independent
cohorts, predefined thresholds, test-retest reliability, and evidence that
score changes correspond to meaningful training or clinical outcomes.

## Reproducibility

The analysis entry point is
[`scripts/build_scirep_analysis.py`](../scripts/build_scirep_analysis.py).
Package versions are pinned in [`requirements.txt`](../requirements.txt).

Random seeds:

| Operation | Seed |
|:--|--:|
| Bootstrap and sampling | 20260618 |
| Classification | 13 |
| Principal component analysis | 20260618 |
| K-means | 20260618 |

Aggregate outputs include:

- [`all_trial_feature_statistics.csv`](../data/derived/all_trial_feature_statistics.csv);
- [`cohort_adjusted_models.csv`](../data/derived/cohort_adjusted_models.csv);
- [`cohort_specific_effects.csv`](../data/derived/cohort_specific_effects.csv);
- [`continuous_performance_validation.csv`](../data/derived/continuous_performance_validation.csv);
- [`kmeans_cluster_summary.csv`](../data/derived/kmeans_cluster_summary.csv);
- [`kmeans_validation.csv`](../data/derived/kmeans_validation.csv);
- [`processing_sensitivity.csv`](../data/derived/processing_sensitivity.csv);
- [`phase_kinematics_summary.csv`](../data/derived/phase_kinematics_summary.csv);
- [`analysis_manifest.json`](../data/derived/analysis_manifest.json).

Participant-level intermediate tables are deliberately omitted until the
complete governed dataset release is available.
