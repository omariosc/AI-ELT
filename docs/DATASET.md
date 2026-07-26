# Dataset guide

## Scope

LASK contains video and instrument measurements from a laparoscopic
peg-transfer task performed in box trainers. A participant uses two graspers to
move objects between pegs. One recording contains one participant attempt.

The collection was designed to support:

- instrument detection, segmentation, and landmark localisation;
- six-channel and seven-channel pose estimation from video;
- instrument tracking through occlusion;
- analysis of motion, coordination, and task timing;
- study of acquisition shift between training settings.

It was not designed as a validated clinical examination. Procedure counts,
career stage, and calculated motion scores should not be interpreted as
interchangeable measures of competence.

## Release stages

The permanent concept DOI is
[10.5281/zenodo.20752650](https://doi.org/10.5281/zenodo.20752650).
[Version 1.0](https://zenodo.org/records/20752651) contains 37 recordings.
Further recordings and phase labels will be added as new versions.

### Public version 1.0

| Cohort | Archive | Role | Recordings |
|:--|:--|:--|--:|
| Paediatric | `DatasetB_BAPES_val.zip` | Validation | 10 |
| Urology 1 | `DatasetC_6DOF_test.zip` | Testing | 8 |
| Urology 2 | `DatasetA_7DOF_train.zip` | Training | 19 |
| **Total** |  |  | **37** |

### Complete collection used in the quantitative analysis

| Cohort | Available | Primary motion set | Phase labelled | Primary phase set |
|:--|--:|--:|--:|--:|
| Paediatric | 30 | 28 | 9 | 9 |
| Urology 1 | 24 | 24 | 10 | 10 |
| Urology 2 | 61 | 55 | 19 | 15 |
| **Total** | **115** | **107** | **38** | **34** |

The primary sets retain study identifiers represented by one recording.
Repeated or multipart recordings remain in the dataset inventory but are not
treated as independent observations in the primary inferential analyses.

## Cohorts

### Paediatric

Data were collected at the British Association of Paediatric Endoscopic
Surgeons meeting in November 2024. Participants were paediatric consultants or
trainees between specialty training years 3 and 7. The questionnaire recorded
lifetime and previous-year skin-to-skin procedure counts. Position,
orientation, and relative jaw-opening channels were recorded at 13 frames per
second.

### Urology 1

Data were collected at a urology simulation boot camp in October 2023.
Fenestrated and curved instruments were tracked at approximately 26 frames per
second. Position and orientation were available, but no jaw-opening channel was
recorded.

### Urology 2

Data were collected at a urology simulation boot camp in October 2024.
Position, orientation, and relative jaw-opening channels were recorded at
13 frames per second.

The trainers, tools, camera settings, coordinate distributions, and insertion
angles differed between cohorts. Raw kinematic values should not be pooled
without cohort adjustment.

Degrees of freedom describe the independently measured motion channels. Six
degrees of freedom comprise three-dimensional position and three-dimensional
orientation. The seventh channel is the relative jaw-opening measurement.

## Archive structure

Each public archive follows this layout:

```text
DatasetX_.../
├── videos/
│   └── <trial>.mp4
├── annotations/
│   └── <trial>.npz
├── DatasetX_..._kinematics.csv
├── participants.csv
└── README.md
```

The videos are H.264 working encodes. Dense kinematics are aligned by frame
number and time. Instrument annotations are sparse, usually sampled at
approximately every 100th frame.

## Kinematic files

The Aurora electromagnetic tracker from Northern Digital Inc. records tool
pose on every sampled frame.

### Position

Position columns ending in `px`, `py`, and `pz` are expressed in millimetres.
The sensor coils are mounted near the instrument base. A fixed calibration
transform estimates the instrument tip, and positions can then be represented
relative to the camera frame.

### Orientation

Orientations are stored as unit quaternions. The seven-channel archives use
`qw`, `qx`, `qy`, and `qz`. The six-channel archive retains its original
`q0`, `q1`, `q2`, and `q3` naming. Check the archive README before converting
between quaternion conventions.

Angular movement in the companion analysis is calculated from the relative
rotation between adjacent unit quaternions. Quaternion signs are aligned before
the optional orientation-smoothing sensitivity analysis because `q` and `-q`
represent the same rotation.

### Jaw opening

The two seven-channel cohorts contain voltage-derived jaw measurements. The
current files may retain `tool1_angle` and `tool2_angle` as historical column
names. These channels should be treated as relative aperture signals unless an
independent physical angle calibration is supplied.

For descriptive analysis, the voltage is low-pass filtered and mapped within
each recording so that the 10th percentile represents closed and the 90th
percentile represents open. Values are clipped to an opening fraction from
zero to one. This removes between-recording voltage offsets but does not
produce an angle in radians or degrees.

### Time

The seven-channel cohorts use seconds. The six-channel archive retains a
millisecond time column. The paper analysis uses fixed cohort rates of
13 frames per second for Paediatric and Urology 2, and 26 frames per second for
Urology 1.

## Instrument annotation files

The `.npz` files are compressed NumPy array archives and can be loaded without
pickle:

```python
import numpy as np

annotation = np.load("annotations/Trial01.npz")
print(annotation.files)
```

Important arrays include:

| Array | Shape | Meaning |
|:--|:--|:--|
| `frame_idx` | `(F,)` | Frame number in the source video and kinematic file |
| `toolN_mask_points` | `(M, 2)` | Concatenated polygon coordinates in pixels |
| `toolN_mask_offsets` | `(F+1,)` | Start and end offsets for each frame polygon |
| `toolN_ee_tip` | `(F, 2)` | Instrument-tip landmark |
| `toolN_ee_left` | `(F, 2)` | Left jaw landmark |
| `toolN_ee_right` | `(F, 2)` | Right jaw landmark |
| `toolN_joint` | `(F, 2)` | End-effector or shaft-joint landmark |
| `toolN_vis_mask` | `(F,)` | Whether a segmentation mask is present |

`N` is 1 or 2. Landmark coordinates are stored as image `(x, y)` positions.
Unavailable landmarks are represented by `NaN`, meaning not a number.

## Participant metadata

`participants.csv` contains one row per released recording:

- dataset and split;
- release trial identifier;
- handedness;
- lifetime laparoscopic procedure count;
- procedure count during the previous year, where collected.

The handedness questionnaire was based on the Edinburgh Handedness Inventory.
The analysis uses the lifetime procedure count as a continuous experience
measure. The historical categories are:

- novice: fewer than 20 procedures;
- intermediate: 20 to 99 procedures;
- expert: at least 100 procedures.

These labels describe procedure-volume strata. They are not expert ratings of
the recorded peg-transfer attempt.

## Dense phase labels

The phase subset includes reach, grasp, transfer, place, nudge, idle, and
dropped-object labels. It is documented separately in
[`PHASES.md`](PHASES.md).

## Known limitations

- The current public release is smaller than the complete collection.
- Cohort and acquisition setting are confounded.
- Jaw opening is absent from Urology 1.
- The relative jaw signal does not provide an absolute physical aperture.
- Fine object possession and grasp quality cannot always be inferred from the
  kinematic channels alone.
- Experience metadata were self-reported.
- Phase labels were produced by one trained annotator.
- Study identifiers cannot exclude the possibility that a participant
  attended more than one event under different identifiers.

These limitations are retained explicitly so that future benchmark results can
be interpreted against the evidence the dataset actually provides.
