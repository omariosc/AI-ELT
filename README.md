# LASK

**LA**paroscopic **S**kill & **K**inematics: a peg-transfer box-trainer dataset
pairing endoscopic video with dense 7-DoF instrument kinematics and manual
instrument annotations.

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20752650-1682D4?style=flat-square)](https://doi.org/10.5281/zenodo.20752650)
[![Download](https://img.shields.io/badge/Download-Zenodo-0F62FE?style=flat-square)](https://doi.org/10.5281/zenodo.20752650)
[![Licence](https://img.shields.io/badge/Licence-CC%20BY%204.0-orange?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)
[![Version](https://img.shields.io/badge/Version-1.0-informational?style=flat-square)](https://doi.org/10.5281/zenodo.20752650)
[![Trials](https://img.shields.io/badge/Trials-37%20annotated-1f4e79?style=flat-square)](#what-is-in-the-dataset)
[![Frames](https://img.shields.io/badge/Frames-~91,000-1f4e79?style=flat-square)](#what-is-in-the-dataset)
[![Paper](https://img.shields.io/badge/Paper-MIUA%202025-548235?style=flat-square)](https://doi.org/10.3389/978-2-8325-5137-0)

> **The dataset is hosted on Zenodo:**
> **[https://doi.org/10.5281/zenodo.20752650](https://doi.org/10.5281/zenodo.20752650)**
>
> This repository is its landing page. Zenodo holds the archived, versioned,
> citable release.

---

## Why it exists

Accurate perception of surgical instruments underpins automated skill assessment
and computer-assisted training in minimally invasive surgery, but public
video-kinematic datasets are scarce, especially for non-in-vivo tasks. LASK
synchronises endoscopic video with ground-truth kinematics for **two graspers
tracked continuously**, so instrument detection, pose estimation and skill
classification can all be benchmarked against the same trials.

## What is in the dataset

**37 annotated trials**, arranged as a cross-cohort benchmark:

| Split | Cohort | Trials | Purpose |
|:--|:--|--:|:--|
| Train | 7-DoF | 19 | In-distribution training |
| Validation | 7-DoF | 10 | In-distribution validation |
| Test | 6-DoF | 8 | Out-of-distribution generalisation |

Spanning surgeons from novice to expert.

**Per-frame kinematics**, roughly 91,000 frames, time-aligned to the video:

- Electromagnetic tool poses: positions in millimetres, orientations as unit quaternions
- Calibrated jaw angles for the 7-DoF cohorts
- Both instruments tracked throughout

**Manual annotations** on labelled keyframes, for both instruments:

- Instrument segmentation masks
- Tooltip and jaw (left/right) keypoints
- Shaft-joint keypoints
- Per-component visibility flags

**Surgeon metadata**: handedness (left, right or both) and procedure experience,
both lifetime and over the last 12 months.

## What it supports

- Multi-class instrument detection and segmentation
- Instrument tracking through occlusion and clutter
- 6-DoF and 7-DoF pose estimation from video alone
- Surgical skill classification
- Cross-configuration generalisation, via the 6-DoF versus 7-DoF split

## Getting the data

Download from the Zenodo record: **[10.5281/zenodo.20752650](https://doi.org/10.5281/zenodo.20752650)**

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), so it
may be used commercially and redistributed provided the dataset is credited.

## Work using LASK

| Work | Venue | What it did |
|:--|:--|:--|
| [7-DoF Laparoscopic Peg Transfer Dataset for Surgical Skill Assessment](https://doi.org/10.3389/978-2-8325-5137-0) | MIUA 2025 | Introduced the dataset. Won Best Presentation at the Doctoral Consortium |
| [Real-Time Tool Detection in Laparoscopic Datasets for Surgical Training in Low-Resource Settings](https://doi.org/10.1049/htl2.70045) | *Healthcare Technology Letters* **12**(1) | Real-time detection on low-cost embedded devices |
| [Bayesian Temporal Pose Networks](https://github.com/omariosc/BTPN) | MICCAI 2026 | Uncertainty-calibrated 7-DoF vision-only pose tracking |

## Citation

Please cite the dataset by its DOI. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff).

```bibtex
@dataset{Choudhry2026LASK,
  title     = {LASK: A Dataset for Laparoscopic Skill and 7-DoF Kinematics},
  author    = {Choudhry, Omar and Jones, Dominic},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0},
  doi       = {10.5281/zenodo.20752650},
  url       = {https://doi.org/10.5281/zenodo.20752650}
}
```

If you use the dataset for skill assessment, please also cite the paper that
introduced it:

```bibtex
@inproceedings{Choudhry2025PegTransfer,
  title     = {7-DoF Laparoscopic Peg Transfer Dataset for Surgical Skill Assessment},
  author    = {Choudhry, Omar and Ali, Sharib and Rajasundaram, Ramanan and
               Biyani, Chandra Shekhar and Jones, Dominic},
  booktitle = {Medical Image Understanding and Analysis (MIUA)},
  publisher = {Frontiers Media SA},
  year      = {2025},
  doi       = {10.3389/978-2-8325-5137-0}
}
```

## Licence

Dataset and the contents of this repository: [CC BY 4.0](LICENSE).

## Contact

[Omar Choudhry](https://omarchoudhry.co.uk), School of Computing, University of
Leeds. Open an issue here with questions about the dataset.
