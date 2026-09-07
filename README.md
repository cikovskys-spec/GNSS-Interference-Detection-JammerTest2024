# GNSS Interference Detection from Receiver-Level Observables

Reproducibility repository for the manuscript on multiclass GNSS interference recognition using receiver-derived observables from the JammerTest 2024 dataset.

## Scope

The repository contains the preprocessing, feature-engineering, window-generation, model-training, and evaluation code used in the study.

The classification task distinguishes:

- spoofing
- jamming
- meaconing

The study uses receiver-level/non-IQ observables derived from:

- NAV-PVT
- RINEX
- MON-RF

## Dataset

The experiments use the JammerTest 2024 dataset.

The processed experiment contains:

- 24 scenarios
- 12 spoofing scenarios
- 7 jamming scenarios
- 5 meaconing scenarios
- 44,639 merged positioning samples
- 2,939 window-level samples
- 217 window-level features

The dataset itself is not redistributed in this repository. Users should obtain the dataset from its official source and place the extracted scenario folders in the location described below.

## Repository structure

```text
GNSS-Interference-Detection-JammerTest2024/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── src/
    └── gnss_interference_detection.py
```

## Software requirements

Python 3.10 or newer is recommended.

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Data layout

After obtaining and extracting the dataset, the expected structure is:

```text
dataset_root/
└── extracted/
    ├── scenario_1/
    │   ├── nav_pvt.csv
    │   ├── rinex.csv
    │   └── mon_rf.csv
    ├── scenario_2/
    │   └── ...
    └── ...
```

The exact scenario folder names are taken from the supplied dataset.

## Running the experiment

Place the final script in:

```text
src/gnss_interference_detection.py
```

Then run:

```bash
python src/gnss_interference_detection.py
```

The pipeline performs:

1. dataset discovery;
2. NAV-PVT, RINEX and MON-RF loading;
3. receiver-level preprocessing;
4. positioning-derived feature extraction;
5. window-level feature generation;
6. scenario-grouped train/test evaluation;
7. ExtraTrees and RandomForest classification;
8. reporting of accuracy, macro-F1, and class-wise precision, recall and F1;
9. generation of numbered figures and result tables.

## Evaluation protocol

The evaluation is performed with repeated scenario-group holdout splits so that samples from the same scenario are not simultaneously used for training and testing.

The reported metrics are aggregated over the repeated splits as mean ± standard deviation.

This evaluation is intentionally different from a conventional random sample-level split because scenario-level grouping reduces the risk of information leakage between training and test data.

## Reproducibility

The repository is intended to provide the complete software needed to reproduce the reported preprocessing, feature engineering, model training and evaluation.

The dataset is not included because it is distributed separately.

For the reported experiment, the main random seed is:

```text
42
```

## Outputs

The pipeline produces:

- numbered figures;
- feature-importance results;
- summary metrics;
- confusion-matrix results;
- other intermediate result tables required by the manuscript.

## Citation

If you use this code, please cite the associated manuscript and the original JammerTest dataset publication/data record.

## Contact

Please use the contact information provided in the associated manuscript for questions concerning the study or repository.
