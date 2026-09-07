# Source code

`gnss_interference_detection.py` is the A1 leakage-safe pipeline used for the
reported manuscript experiment.

The script contains:

- NAV-PVT, RINEX and MON-RF loading;
- receiver-level preprocessing;
- causal positioning-derived features;
- training-only integrity-risk normalization;
- temporal window generation;
- repeated scenario-group evaluation;
- ExtraTrees and RandomForest models;
- class-wise precision, recall and F1 reporting;
- manuscript figures and summary outputs.

The file is intentionally kept as the experiment script rather than reduced
to a simplified example, so that the repository corresponds to the reported
analysis.
