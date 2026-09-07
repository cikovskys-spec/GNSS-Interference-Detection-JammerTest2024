# ============================================================
# FINAL REVISED GNSS POSITIONING INTEGRITY PIPELINE
# A1: LEAKAGE-SAFE PREPROCESSING
# Real dataset: JammerTest 2024
# ============================================================

import os, re, glob, tarfile, shutil, math, random, warnings, requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.ndimage import uniform_filter1d

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from PIL import Image, ImageOps, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE = "/content/gnss_final_q1_positioning"
OUT = "/content/gnss_final_q1_positioning_outputs"
FIG_DIR = f"{OUT}/figures_numbered"
PANEL_DIR = f"{OUT}/figure_panels_4"
TRAJ_DIR = f"{OUT}/trajectory_figures"

for d in [BASE, OUT, FIG_DIR, PANEL_DIR, TRAJ_DIR]:
    os.makedirs(d, exist_ok=True)

print("Output folder:", OUT)


# ============================================================
# 1. DOWNLOAD REAL JAMMERTEST 2024 DATASET
# ============================================================

ZENODO_URL = (
    "https://zenodo.org/records/15911589/files/"
    "GNSS_DATASET_JAMMING_SPOOFING.tar.gz?download=1"
)

TAR_PATH = f"{BASE}/GNSS_DATASET_JAMMING_SPOOFING.tar.gz"
EXTRACT_DIR = f"{BASE}/extracted"

if not os.path.exists(TAR_PATH):
    print("Downloading JammerTest 2024 dataset...")

    r = requests.get(ZENODO_URL, stream=True)

    total = int(r.headers.get("content-length", 0))

    with open(TAR_PATH, "wb") as f:
        for chunk in tqdm(
            r.iter_content(chunk_size=1024 * 1024),
            total=max(1, total // (1024 * 1024))
        ):
            if chunk:
                f.write(chunk)
else:
    print("Archive already downloaded.")


if not os.path.exists(EXTRACT_DIR):
    print("Extracting dataset...")

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)
else:
    print("Already extracted.")


print("Dataset root:", EXTRACT_DIR)


# ============================================================
# 2. DISCOVER SCENARIOS
# ============================================================

def infer_attack_label(path):

    p = path.lower()

    if "spoof" in p:
        return "spoofing"

    if "jam" in p:
        return "jamming"

    if "meacon" in p:
        return "meaconing"

    if "combined" in p or "mixed" in p:
        return "combined"

    if "nominal" in p or "clean" in p or "reference" in p:
        return "nominal"

    return "unknown"


def infer_motion_type(path):

    p = path.lower()

    if "dynamic" in p:
        return "dynamic"

    if "stationary" in p or "static" in p:
        return "stationary"

    return "unknown"


def infer_power_level(path):

    p = path.lower()

    if "low" in p:
        return "low"

    if "medium" in p:
        return "medium"

    if "high" in p:
        return "high"

    return "unknown"


def infer_band_setup(path):

    p = path.lower()

    m = re.search(r"bands?[_\-\s]*([^/\\]+)", p)

    if m:
        return m.group(1)

    return "unknown"


nav_files = sorted(
    glob.glob(
        EXTRACT_DIR + "/**/nav_pvt.csv",
        recursive=True
    )
)

print("NAV-PVT files:", len(nav_files))


scenario_rows = []

for nav in nav_files:

    folder = os.path.dirname(nav)

    scenario_rows.append({
        "scenario_id": len(scenario_rows),
        "folder": folder,
        "nav_file": nav,
        "rinex_file": os.path.join(
            folder,
            "rinex.csv"
        ),
        "monrf_file": os.path.join(
            folder,
            "mon_rf.csv"
        ),
        "attack_label": infer_attack_label(nav),
        "motion_type": infer_motion_type(nav),
        "power_level": infer_power_level(nav),
        "band_setup": infer_band_setup(nav)
    })


scenario_table = pd.DataFrame(scenario_rows)

scenario_table.to_csv(
    f"{OUT}/scenario_table.csv",
    index=False
)

print(
    scenario_table["attack_label"].value_counts()
)


if len(scenario_table) == 0:
    raise RuntimeError(
        "No nav_pvt.csv files found."
    )


# ============================================================
# 3. LOAD NAV-PVT + RINEX + MON-RF
# ============================================================

def safe_read_csv(path):

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception:

        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python"
            )

        except Exception:
            return pd.DataFrame()


def clean_nav_pvt(nav):

    df = safe_read_csv(nav)

    if (
        len(df) == 0
        or "lat" not in df.columns
        or "lon" not in df.columns
    ):
        return pd.DataFrame()

    out = pd.DataFrame()

    out["sample_id"] = np.arange(len(df))

    cols = [
        "lat",
        "lon",
        "height",
        "hMSL",
        "hAcc",
        "vAcc",
        "gSpeed",
        "headMot",
        "headAcc",
        "numSV",
        "fixType",
        "pDOP",
        "sAcc",
        "tAcc",
        "velN",
        "velE",
        "velD",
        "iTOW"
    ]

    for c in cols:

        if c in df.columns:

            out[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

        else:
            out[c] = np.nan


    if "real_time" in df.columns:

        out["real_time"] = pd.to_datetime(
            df["real_time"],
            errors="coerce"
        )

    else:
        out["real_time"] = pd.NaT


    out = out.dropna(
        subset=["lat", "lon"]
    )

    out = out[
        out["lat"].between(-90, 90)
        &
        out["lon"].between(-180, 180)
    ]


    if len(out) >= 20:
        return out.reset_index(drop=True)

    return pd.DataFrame()


def aggregate_rinex(rinex_file):

    df = safe_read_csv(rinex_file)

    if (
        len(df) == 0
        or "time" not in df.columns
    ):
        return pd.DataFrame()


    df["real_time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["real_time"]
    )


    num_cols = [
        c
        for c in df.columns
        if c not in [
            "time",
            "satellite",
            "real_time"
        ]
    ]


    for c in num_cols:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )


    agg_dict = {}

    for c in num_cols:

        cl = c.lower()

        if any(
            k in cl
            for k in [
                "snr",
                "doppler",
                "pseudorange",
                "carrier"
            ]
        ):
            agg_dict[c] = [
                "mean",
                "std",
                "min",
                "max"
            ]


    if len(agg_dict) == 0:
        return pd.DataFrame()


    g = (
        df
        .groupby("real_time")
        .agg(agg_dict)
    )


    g.columns = [
        "rinex_" + "_".join(col).strip()
        for col in g.columns.values
    ]

    g = g.reset_index()


    if "satellite" in df.columns:

        sat_count = (
            df
            .groupby("real_time")["satellite"]
            .nunique()
            .reset_index()
        )

        sat_count.columns = [
            "real_time",
            "rinex_satellite_count"
        ]

        g = g.merge(
            sat_count,
            on="real_time",
            how="left"
        )


    return g


def aggregate_monrf(monrf_file):

    df = safe_read_csv(monrf_file)

    if (
        len(df) == 0
        or "real_time" not in df.columns
    ):
        return pd.DataFrame()


    df["real_time"] = pd.to_datetime(
        df["real_time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["real_time"]
    )


    candidate_cols = []

    for c in df.columns:

        cl = c.lower()

        if any(
            k in cl
            for k in [
                "agccnt",
                "jamind",
                "jammingstate",
                "noiseperms",
                "magi",
                "magq",
                "ofsi",
                "ofsq"
            ]
        ):
            candidate_cols.append(c)


    for c in candidate_cols:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )


    if len(candidate_cols) == 0:
        return pd.DataFrame()


    g = (
        df
        .groupby("real_time")[candidate_cols]
        .agg([
            "mean",
            "std",
            "min",
            "max"
        ])
    )


    g.columns = [
        "monrf_" + "_".join(col).strip()
        for col in g.columns.values
    ]

    g = g.reset_index()

    return g


def merge_asof_time(
    nav,
    rinex,
    monrf
):

    nav = (
        nav
        .sort_values(
            "real_time"
        )
        .copy()
    )


    if nav["real_time"].notna().sum() == 0:
        return nav


    merged = nav.copy()


    if len(rinex) > 0:

        rinex = (
            rinex
            .sort_values("real_time")
        )

        merged = pd.merge_asof(
            merged,
            rinex,
            on="real_time",
            direction="nearest",
            tolerance=pd.Timedelta(
                "1500ms"
            )
        )


    if len(monrf) > 0:

        monrf = (
            monrf
            .sort_values("real_time")
        )

        merged = pd.merge_asof(
            merged,
            monrf,
            on="real_time",
            direction="nearest",
            tolerance=pd.Timedelta(
                "1500ms"
            )
        )


    return merged


all_scenarios = []


for _, row in tqdm(
    scenario_table.iterrows(),
    total=len(scenario_table)
):

    nav = clean_nav_pvt(
        row["nav_file"]
    )

    if len(nav) == 0:
        continue


    rinex = aggregate_rinex(
        row["rinex_file"]
    )

    monrf = aggregate_monrf(
        row["monrf_file"]
    )


    merged = merge_asof_time(
        nav,
        rinex,
        monrf
    )


    merged["scenario_id"] = (
        row["scenario_id"]
    )

    merged["source_file"] = (
        row["nav_file"]
    )

    merged["folder"] = (
        row["folder"]
    )

    merged["attack_label"] = (
        row["attack_label"]
    )

    merged["motion_type"] = (
        row["motion_type"]
    )

    merged["power_level"] = (
        row["power_level"]
    )

    merged["band_setup"] = (
        row["band_setup"]
    )


    all_scenarios.append(
        merged
    )


if len(all_scenarios) == 0:
    raise RuntimeError(
        "No valid scenarios were loaded."
    )


data = (
    pd.concat(
        all_scenarios,
        ignore_index=True
    )
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


print(
    "Merged samples:",
    len(data)
)

print(
    "Merged scenarios:",
    data["scenario_id"].nunique()
)

print(
    data["attack_label"].value_counts()
)


data.to_csv(
    f"{OUT}/merged_gnss_positioning_dataset.csv",
    index=False
)


# ============================================================
# 4. POSITIONING IMPACT FEATURES
# ============================================================
#
# A1 LEAKAGE-SAFE CHANGES:
#
# 1. The scenario-wide median position is NOT used.
#    The first valid receiver position is used as the reference.
#
# 2. The centered moving-average filter is NOT used.
#    A causal rolling mean uses only current and previous samples.
#
# 3. Height deviation is referenced to the first valid height,
#    not the scenario-wide median.
#
# 4. Missing-value imputation and percentile normalization used
#    for the ML integrity-risk feature are fitted inside each
#    train/test split.
# ============================================================


def add_local_coordinates(g):

    g = (
        g
        .copy()
        .sort_values(
            "real_time",
            kind="stable"
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # CAUSAL SCENARIO REFERENCE
    # --------------------------------------------------------
    #
    # First valid position is known at the beginning of the
    # scenario and therefore does not require future observations.
    #

    lat0 = g["lat"].iloc[0]
    lon0 = g["lon"].iloc[0]


    g["east_m"] = (
        (g["lon"] - lon0)
        * 111320
        * np.cos(
            np.deg2rad(lat0)
        )
    )


    g["north_m"] = (
        (g["lat"] - lat0)
        * 110540
    )


    g["radial_drift_m"] = np.sqrt(
        g["east_m"]**2
        +
        g["north_m"]**2
    )


    # --------------------------------------------------------
    # CAUSAL LOCAL SMOOTHING
    # --------------------------------------------------------
    #
    # Only current and previous observations are used.
    #

    g["east_smooth_m"] = (
        pd.Series(
            g["east_m"].values
        )
        .rolling(
            window=9,
            min_periods=1
        )
        .mean()
        .values
    )


    g["north_smooth_m"] = (
        pd.Series(
            g["north_m"].values
        )
        .rolling(
            window=9,
            min_periods=1
        )
        .mean()
        .values
    )


    g["position_error_proxy_m"] = np.sqrt(
        (
            g["east_m"]
            -
            g["east_smooth_m"]
        )**2
        +
        (
            g["north_m"]
            -
            g["north_smooth_m"]
        )**2
    )


    g["step_distance_m"] = np.r_[
        0,
        np.sqrt(
            np.diff(g["east_m"])**2
            +
            np.diff(g["north_m"])**2
        )
    ]


    # --------------------------------------------------------
    # CAUSAL HEIGHT REFERENCE
    # --------------------------------------------------------

    if g["height"].notna().any():

        first_height = (
            g["height"]
            .dropna()
            .iloc[0]
        )

    else:
        first_height = 0.0


    g["height_dev_m"] = (
        g["height"]
        -
        first_height
    )

    g["height_abs_dev_m"] = (
        np.abs(
            g["height_dev_m"]
        )
    )


    # --------------------------------------------------------
    # HORIZONTAL VELOCITY
    # --------------------------------------------------------

    velN = pd.to_numeric(
        g["velN"],
        errors="coerce"
    )

    velE = pd.to_numeric(
        g["velE"],
        errors="coerce"
    )


    g["velocity_horizontal_mps"] = np.sqrt(
        velN**2 + velE**2
    )


    # --------------------------------------------------------
    # ACCELERATION PROXY
    # --------------------------------------------------------

    velocity_values = (
        g["velocity_horizontal_mps"]
        .to_numpy(
            dtype=float
        )
    )


    if len(velocity_values) > 0:

        g["accel_proxy"] = np.r_[
            0,
            np.abs(
                np.diff(
                    velocity_values
                )
            )
        ]

    else:

        g["accel_proxy"] = []


    return g


data2 = pd.concat(
    [
        add_local_coordinates(g)
        for _, g in data.groupby(
            "scenario_id"
        )
    ],
    ignore_index=True
)


base_cols = [
    "hAcc",
    "vAcc",
    "pDOP",
    "sAcc",
    "tAcc",
    "gSpeed",
    "numSV",
    "fixType",
    "radial_drift_m",
    "position_error_proxy_m",
    "step_distance_m",
    "height_abs_dev_m",
    "velocity_horizontal_mps",
    "accel_proxy"
]


for c in base_cols:

    data2[c] = pd.to_numeric(
        data2[c],
        errors="coerce"
    )


# ============================================================
# 4A. TRAINING-ONLY INTEGRITY-RISK NORMALIZATION
# ============================================================

def fit_risk_parameters(train_df):
    """
    Fit all parameters required by the integrity-risk index
    using TRAINING DATA ONLY.

    The fitted parameters consist of:
    - training median for missing-value replacement
    - training 5th percentile
    - training 95th percentile

    These parameters can then be applied to both training
    and test data without using any test-set statistics.
    """

    params = {}


    risk_columns = [
        "hAcc",
        "radial_drift_m",
        "position_error_proxy_m",
        "step_distance_m",
        "pDOP",
        "height_abs_dev_m"
    ]


    for c in risk_columns:

        s = pd.to_numeric(
            train_df[c],
            errors="coerce"
        )


        s = (
            s
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )


        med = s.median()


        if not np.isfinite(med):
            med = 0.0


        s_filled = s.fillna(
            med
        )


        lo = np.nanpercentile(
            s_filled,
            5
        )

        hi = np.nanpercentile(
            s_filled,
            95
        )


        if not np.isfinite(lo):
            lo = 0.0


        if not np.isfinite(hi):
            hi = lo


        params[c] = {
            "median": float(med),
            "p05": float(lo),
            "p95": float(hi)
        }


    return params


def apply_risk_parameters(
    df,
    params
):
    """
    Apply parameters fitted from training data.

    No statistics are calculated from the dataframe passed
    to this function.
    """

    out = df.copy()


    weights = {
        "hAcc": 0.25,
        "radial_drift_m": 0.20,
        "position_error_proxy_m": 0.20,
        "step_distance_m": 0.15,
        "pDOP": 0.10,
        "height_abs_dev_m": 0.10
    }


    risk = np.zeros(
        len(out),
        dtype=float
    )


    for c, w in weights.items():

        s = pd.to_numeric(
            out[c],
            errors="coerce"
        )


        s = (
            s
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )


        med = params[c]["median"]
        lo = params[c]["p05"]
        hi = params[c]["p95"]


        s = s.fillna(
            med
        )


        norm = np.clip(
            (
                s - lo
            )
            /
            (
                hi - lo
                +
                1e-9
            ),
            0,
            1
        )


        risk += (
            w
            *
            norm.to_numpy()
        )


    out["integrity_risk_index"] = risk


    out["operational_state"] = pd.cut(
        out["integrity_risk_index"],
        bins=[
            -0.01,
            0.33,
            0.66,
            1.01
        ],
        labels=[
            "low-risk",
            "degraded",
            "unsafe"
        ]
    ).astype(str)


    return out


# ============================================================
# DESCRIPTIVE RISK INDEX
# ============================================================
#
# This full-dataset version is retained only for descriptive
# figures and statistics.
#
# IMPORTANT:
# It is NOT passed directly to the ML classifier.
#
# The ML version is recalculated independently inside every
# train/test split using training-only parameters.
# ============================================================

descriptive_risk_params = fit_risk_parameters(
    data2
)


data2 = apply_risk_parameters(
    data2,
    descriptive_risk_params
)


data2.to_csv(
    f"{OUT}/positioning_integrity_features.csv",
    index=False
)


print(
    "Final rows:",
    len(data2)
)

print(
    "Operational states:"
)

print(
    data2["operational_state"]
    .value_counts()
)


# ============================================================
# 5. FIGURE HELPERS
# ============================================================

figure_registry = []

fig_counter = 1


def save_numbered_figure(
    title_slug
):

    global fig_counter

    fname = (
        f"Figure_{fig_counter:02d}_"
        f"{title_slug}.png"
    )

    path = os.path.join(
        FIG_DIR,
        fname
    )


    plt.tight_layout()


    plt.savefig(
        path,
        dpi=600,
        bbox_inches="tight"
    )


    plt.close()


    figure_registry.append(
        path
    )


    print(
        "Saved:",
        path
    )


    fig_counter += 1


def robust_clip_trace(
    g,
    q=0.995
):

    g = g.copy()

    r = np.sqrt(
        g["east_m"]**2
        +
        g["north_m"]**2
    )


    lim = np.nanquantile(
        r,
        q
    )


    return g[
        r <= lim
    ]


def common_axis_limit(
    df,
    q=0.995
):

    r = np.sqrt(
        df["east_m"]**2
        +
        df["north_m"]**2
    )


    lim = np.nanquantile(
        r,
        q
    )


    if (
        not np.isfinite(lim)
        or lim == 0
    ):
        lim = 1e-3


    return lim * 1.10


def select_representative_scenario(
    df,
    label
):

    candidates = []


    for sid, g in (
        df[
            df["attack_label"] == label
        ]
        .groupby("scenario_id")
    ):

        score = np.nanpercentile(
            g["radial_drift_m"],
            95
        )

        candidates.append(
            (score, sid)
        )


    if len(candidates) == 0:
        return None


    return sorted(
        candidates,
        reverse=True
    )[0][1]


# ============================================================
# 6. FINAL NUMBERED ARTICLE FIGURES
# ============================================================

# ------------------------------------------------------------
# Figure 1
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)


counts = (
    data2["attack_label"]
    .value_counts()
)


plt.bar(
    counts.index,
    counts.values
)


plt.ylabel(
    "Number of positioning samples"
)


plt.title(
    "Figure 1. Dataset composition by interference type"
)


plt.grid(
    axis="y",
    alpha=0.3
)


save_numbered_figure(
    "dataset_composition_by_interference_type"
)


# ------------------------------------------------------------
# Figures 2-4
# Representative trajectories
# ------------------------------------------------------------

for label in [
    "jamming",
    "meaconing",
    "spoofing"
]:

    sid = select_representative_scenario(
        data2,
        label
    )


    if sid is None:
        continue


    g = (
        data2[
            data2["scenario_id"] == sid
        ]
        .copy()
        .reset_index(drop=True)
    )


    g = robust_clip_trace(
        g,
        q=0.995
    )


    plt.figure(
        figsize=(8, 7)
    )


    t = np.linspace(
        0,
        1,
        len(g)
    )


    plt.scatter(
        g["east_m"],
        g["north_m"],
        c=t,
        s=16,
        alpha=0.85
    )


    plt.plot(
        g["east_m"],
        g["north_m"],
        linewidth=1.2,
        alpha=0.45
    )


    plt.scatter(
        g["east_m"].iloc[0],
        g["north_m"].iloc[0],
        marker="o",
        s=110,
        label="Start"
    )


    plt.scatter(
        g["east_m"].iloc[-1],
        g["north_m"].iloc[-1],
        marker="X",
        s=130,
        label="End"
    )


    local_lim = common_axis_limit(
        g,
        q=0.995
    )


    plt.xlim(
        -local_lim,
        local_lim
    )


    plt.ylim(
        -local_lim,
        local_lim
    )


    plt.gca().set_aspect(
        "equal",
        adjustable="box"
    )


    plt.xlabel(
        "Relative East displacement [m]"
    )


    plt.ylabel(
        "Relative North displacement [m]"
    )


    plt.title(
        f"Figure {fig_counter}. "
        f"Representative GNSS trajectory degradation: "
        f"{label}"
    )


    plt.legend()


    plt.grid(
        True,
        alpha=0.3
    )


    cbar = plt.colorbar()

    cbar.set_label(
        "Normalized time"
    )


    save_numbered_figure(
        f"representative_trajectory_degradation_{label}"
    )


# ------------------------------------------------------------
# Figure 5
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 4)
)


for label, g in data2.groupby(
    "attack_label"
):

    gg = (
        g
        .reset_index(drop=True)
    )


    plt.plot(
        gg.index,
        np.log10(
            gg[
                "position_error_proxy_m"
            ]
            + 1e-9
        ),
        linewidth=0.8,
        alpha=0.8,
        label=label
    )


plt.xlabel(
    "Sample index"
)


plt.ylabel(
    "log10(position error proxy + ε)"
)


plt.title(
    "Figure 5. Local position instability under interference"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


save_numbered_figure(
    "local_position_instability_log_scale"
)


# ------------------------------------------------------------
# Figure 6
# ------------------------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(11, 8),
    sharex=True
)


categories = [
    "jamming",
    "meaconing",
    "spoofing"
]


for ax, label in zip(
    axes,
    categories
):

    subset = (
        data2[
            data2["attack_label"] == label
        ]
        .reset_index(drop=True)
    )


    ax.plot(
        subset.index,
        subset["hAcc"],
        label=label.capitalize()
    )


    ax.set_ylabel(
        "hAcc [m]"
    )


    ax.set_title(
        f"Receiver Horizontal Accuracy: "
        f"{label.capitalize()}"
    )


    ax.grid(
        True,
        alpha=0.3
    )


    ax.legend(
        loc="upper right"
    )


axes[-1].set_xlabel(
    "Sample index"
)


fig.suptitle(
    "Figure 6. Receiver-reported horizontal accuracy under interference",
    fontsize=12
)


save_numbered_figure(
    "receiver_horizontal_accuracy_under_interference"
)


# ------------------------------------------------------------
# Figure 7
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 4)
)


for label, g in data2.groupby(
    "attack_label"
):

    gg = (
        g
        .reset_index(drop=True)
    )


    plt.plot(
        gg.index,
        gg["integrity_risk_index"],
        linewidth=0.8,
        alpha=0.8,
        label=label
    )


plt.xlabel(
    "Sample index"
)


plt.ylabel(
    "Integrity risk index"
)


plt.title(
    "Figure 7. Operational GNSS integrity risk over time"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


save_numbered_figure(
    "operational_integrity_risk_over_time"
)


# ------------------------------------------------------------
# Figure 8
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)


for label, g in data2.groupby(
    "attack_label"
):

    vals = np.sort(
        g[
            "radial_drift_m"
        ]
        .dropna()
        .values
    )


    vals = vals[
        np.isfinite(vals)
    ]


    cdf = (
        np.arange(
            1,
            len(vals) + 1
        )
        /
        len(vals)
    )


    plt.plot(
        vals,
        cdf,
        linewidth=2,
        label=label
    )


plt.xlabel(
    "Radial drift [m]"
)


plt.ylabel(
    "CDF"
)


plt.title(
    "Figure 8. CDF of GNSS position drift"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


save_numbered_figure(
    "cdf_position_drift"
)


# ------------------------------------------------------------
# Figure 9
# ------------------------------------------------------------

order = [
    "jamming",
    "meaconing",
    "spoofing"
]


vals = []
labels = []


for label in order:

    if label in data2[
        "attack_label"
    ].unique():

        v = (
            data2.loc[
                data2["attack_label"] == label,
                "radial_drift_m"
            ]
            .dropna()
            .values
        )


        v = v[
            np.isfinite(v)
        ]


        vals.append(v)
        labels.append(label)


plt.figure(
    figsize=(8, 5.5)
)


plt.violinplot(
    vals,
    showmeans=False,
    showmedians=True,
    showextrema=False
)


plt.boxplot(
    vals,
    labels=labels,
    widths=0.18,
    showfliers=False
)


plt.ylabel(
    "Radial drift [m]"
)


plt.ylim(
    0.00,
    0.10
)


plt.title(
    "Figure 9. Distribution of GNSS position drift by interference type"
)


plt.grid(
    True,
    axis="y",
    alpha=0.3
)


for i, v in enumerate(
    vals,
    start=1
):

    med = np.median(v)

    p95 = np.percentile(
        v,
        95
    )


    plt.text(
        i,
        min(
            p95,
            0.095
        ),
        f"95%={p95:.3g} m",
        ha="center",
        va="bottom",
        fontsize=8
    )


    plt.text(
        i,
        min(
            med,
            0.090
        ),
        f"med={med:.3g}",
        ha="center",
        va="top",
        fontsize=8
    )


save_numbered_figure(
    "position_drift_distribution_violin_box"
)


# ------------------------------------------------------------
# Figure 10
# ------------------------------------------------------------

heat_cols = [
    "radial_drift_m",
    "position_error_proxy_m",
    "step_distance_m",
    "hAcc",
    "vAcc",
    "pDOP",
    "gSpeed",
    "height_abs_dev_m",
    "integrity_risk_index"
]


sample_sid = (
    data2
    .groupby("scenario_id")
    .size()
    .sort_values(
        ascending=False
    )
    .index[0]
)


sample = (
    data2[
        data2["scenario_id"] == sample_sid
    ]
    .reset_index(drop=True)
)


heat = sample[
    heat_cols
].copy()


for c in heat.columns:

    heat[c] = pd.to_numeric(
        heat[c],
        errors="coerce"
    )


    heat[c] = heat[c].fillna(
        heat[c].median()
    )


    heat[c] = np.log10(
        np.abs(
            heat[c]
        )
        +
        1e-9
    )


    heat[c] = (
        heat[c]
        -
        heat[c].min()
    ) / (
        heat[c].max()
        -
        heat[c].min()
        +
        1e-9
    )


plt.figure(
    figsize=(12, 4)
)


plt.imshow(
    heat.T,
    aspect="auto",
    origin="lower"
)


plt.yticks(
    range(len(heat_cols)),
    heat_cols
)


plt.xlabel(
    "Sample index"
)


plt.title(
    "Figure 10. Normalized positioning integrity feature heatmap"
)


plt.colorbar(
    label="Normalized log-scaled value"
)


save_numbered_figure(
    "normalized_positioning_integrity_heatmap"
)


# ------------------------------------------------------------
# Figure 11
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)


state_counts = (
    data2["operational_state"]
    .value_counts()
    .reindex(
        [
            "low-risk",
            "degraded",
            "unsafe"
        ]
    )
    .dropna()
)


plt.bar(
    state_counts.index,
    state_counts.values
)


plt.ylabel(
    "Number of samples"
)


plt.title(
    "Figure 11. Operational integrity state distribution"
)


plt.grid(
    axis="y",
    alpha=0.3
)


save_numbered_figure(
    "operational_integrity_state_distribution"
)


# ============================================================
# 7. WINDOW-LEVEL LEAKAGE-SAFE ML
# ============================================================
#
# A1 principle:
#
# Base windows are created without the integrity-risk index.
# Scenario groups are then split.
#
# For every split:
#
#   1. Training scenarios are identified.
#   2. Integrity-risk medians and percentiles are fitted ONLY
#      on training scenarios.
#   3. The fitted parameters are applied to both train and test.
#   4. Windows are generated separately for train and test.
#   5. SimpleImputer is fitted only on X_train.
#   6. StandardScaler is fitted only on X_train.
#   7. The classifier is fitted only on X_train.
#
# Therefore no test-set feature distribution is used to fit
# preprocessing parameters.
# ============================================================


window_feature_cols_base = [
    "radial_drift_m",
    "position_error_proxy_m",
    "step_distance_m",
    "hAcc",
    "vAcc",
    "pDOP",
    "sAcc",
    "tAcc",
    "gSpeed",
    "height_abs_dev_m",
    "velocity_horizontal_mps",
    "accel_proxy"
]


# Receiver-derived RINEX/MON-RF columns.
#
# Their availability is determined from the existing data schema.
# No target information is used here.
#

extra_cols = [
    c
    for c in data2.columns
    if (
        c.startswith("rinex_")
        or
        c.startswith("monrf_")
    )
]


extra_cols = extra_cols[:30]


window_feature_cols = (
    window_feature_cols_base
    +
    extra_cols
)


def slope_feature(x):

    x = np.asarray(
        x,
        dtype=float
    )


    valid = np.isfinite(x)


    if valid.sum() < 2:
        return 0.0


    t = np.arange(
        len(x),
        dtype=float
    )


    try:

        return np.polyfit(
            t[valid],
            x[valid],
            1
        )[0]

    except Exception:

        return 0.0


def make_windows(
    df,
    win=30,
    step=15,
    include_integrity_risk=False
):

    """
    Create temporal windows.

    The integrity-risk feature is optional because the ML version
    must be calculated after training-only risk normalization
    has been fitted for the current split.
    """

    rows = []


    feature_cols = (
        list(window_feature_cols_base)
        +
        list(extra_cols)
    )


    if include_integrity_risk:

        feature_cols = (
            feature_cols
            +
            ["integrity_risk_index"]
        )


    for sid, g in df.groupby(
        "scenario_id"
    ):

        g = (
            g
            .sort_values(
                "real_time",
                kind="stable"
            )
            .reset_index(drop=True)
        )


        if len(g) < win:
            continue


        for start in range(
            0,
            len(g) - win + 1,
            step
        ):

            w = g.iloc[
                start:start + win
            ]


            row = {
                "scenario_id": sid,
                "attack_label": (
                    w[
                        "attack_label"
                    ].iloc[0]
                ),
                "motion_type": (
                    w[
                        "motion_type"
                    ].iloc[0]
                ),
                "power_level": (
                    w[
                        "power_level"
                    ].iloc[0]
                ),
                "band_setup": (
                    w[
                        "band_setup"
                    ].iloc[0]
                ),
                "window_start": start,
                "window_end": (
                    start + win
                )
            }


            for c in feature_cols:

                arr = pd.to_numeric(
                    w[c],
                    errors="coerce"
                ).to_numpy(
                    dtype=float
                )


                if np.isfinite(
                    arr
                ).any():

                    row[
                        f"{c}_mean"
                    ] = np.nanmean(
                        arr
                    )


                    row[
                        f"{c}_std"
                    ] = np.nanstd(
                        arr
                    )


                    row[
                        f"{c}_min"
                    ] = np.nanmin(
                        arr
                    )


                    row[
                        f"{c}_max"
                    ] = np.nanmax(
                        arr
                    )


                    row[
                        f"{c}_slope"
                    ] = slope_feature(
                        arr
                    )

                else:

                    row[
                        f"{c}_mean"
                    ] = np.nan


                    row[
                        f"{c}_std"
                    ] = np.nan


                    row[
                        f"{c}_min"
                    ] = np.nan


                    row[
                        f"{c}_max"
                    ] = np.nan


                    row[
                        f"{c}_slope"
                    ] = 0.0


            rows.append(row)


    return pd.DataFrame(
        rows
    )


# ------------------------------------------------------------
# BASE WINDOWS WITHOUT INTEGRITY-RISK FEATURE
# ------------------------------------------------------------

windows_base = make_windows(
    data2,
    win=30,
    step=15,
    include_integrity_risk=False
)


windows_base = (
    windows_base
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna(
        axis=1,
        how="all"
    )
)


windows_base.to_csv(
    f"{OUT}/window_level_dataset_base_features.csv",
    index=False
)


print(
    "Base window dataset:",
    windows_base.shape
)


print(
    windows_base[
        "attack_label"
    ].value_counts()
)


# ------------------------------------------------------------
# BASE ML FEATURE COLUMNS
# ------------------------------------------------------------

base_ml_feature_cols = [
    c
    for c in windows_base.columns
    if any(
        c.endswith(s)
        for s in [
            "_mean",
            "_std",
            "_min",
            "_max",
            "_slope"
        ]
    )
]


# Target encoding does not use feature information.
le = LabelEncoder()

le.fit(
    windows_base[
        "attack_label"
    ]
)


y_base = le.transform(
    windows_base[
        "attack_label"
    ]
)


groups_base = (
    windows_base[
        "scenario_id"
    ].values
)


# ============================================================
# REPEATED SCENARIO-GROUP EVALUATION
# ============================================================

gss = GroupShuffleSplit(
    n_splits=10,
    test_size=0.30,
    random_state=SEED
)


models = {

    "ExtraTrees":
        ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            max_features="sqrt",
            criterion="gini",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        )
}


print(
    "\n=== Running repeated scenario-group evaluation ==="
)


eval_results = {}

first_split_outputs = {}


for m_name, clf in models.items():

    accs = []
    macro_f1s = []


    precisions = {
        c: []
        for c in le.classes_
    }


    recalls = {
        c: []
        for c in le.classes_
    }


    f1s = {
        c: []
        for c in le.classes_
    }


    supports = {
        c: []
        for c in le.classes_
    }


    first_split_saved = False


    # --------------------------------------------------------
    # REPEATED GROUP SPLITS
    # --------------------------------------------------------

    for split_no, (
        train_idx,
        test_idx
    ) in enumerate(
        gss.split(
            windows_base,
            y_base,
            groups=groups_base
        ),
        start=1
    ):


        # ----------------------------------------------------
        # IDENTIFY TRAINING AND TESTING SCENARIOS
        # ----------------------------------------------------

        train_sids = set(
            windows_base
            .iloc[
                train_idx
            ][
                "scenario_id"
            ]
            .unique()
        )


        test_sids = set(
            windows_base
            .iloc[
                test_idx
            ][
                "scenario_id"
            ]
            .unique()
        )


        # ----------------------------------------------------
        # SAMPLE-LEVEL DATA FOR THIS FOLD
        # ----------------------------------------------------

        train_sample_df = (
            data2[
                data2[
                    "scenario_id"
                ].isin(
                    train_sids
                )
            ]
            .copy()
        )


        test_sample_df = (
            data2[
                data2[
                    "scenario_id"
                ].isin(
                    test_sids
                )
            ]
            .copy()
        )


        # ----------------------------------------------------
        # FIT INTEGRITY-RISK PARAMETERS
        # TRAINING DATA ONLY
        # ----------------------------------------------------

        fold_risk_params = (
            fit_risk_parameters(
                train_sample_df
            )
        )


        # ----------------------------------------------------
        # APPLY TRAINING-ONLY PARAMETERS
        # ----------------------------------------------------

        train_sample_df = (
            apply_risk_parameters(
                train_sample_df,
                fold_risk_params
            )
        )


        test_sample_df = (
            apply_risk_parameters(
                test_sample_df,
                fold_risk_params
            )
        )


        # ----------------------------------------------------
        # CREATE TRAINING WINDOWS
        # ----------------------------------------------------

        train_windows = make_windows(
            train_sample_df,
            win=30,
            step=15,
            include_integrity_risk=True
        )


        # ----------------------------------------------------
        # CREATE TEST WINDOWS
        # ----------------------------------------------------

        test_windows = make_windows(
            test_sample_df,
            win=30,
            step=15,
            include_integrity_risk=True
        )


        # ----------------------------------------------------
        # IDENTICAL FEATURE SCHEMA
        # ----------------------------------------------------

        fold_ml_feature_cols = [
            c
            for c in train_windows.columns
            if any(
                c.endswith(s)
                for s in [
                    "_mean",
                    "_std",
                    "_min",
                    "_max",
                    "_slope"
                ]
            )
        ]


        fold_ml_feature_cols = [
            c
            for c in fold_ml_feature_cols
            if c in test_windows.columns
        ]


        # ----------------------------------------------------
        # TRAINING FEATURES
        # ----------------------------------------------------

        X_tr = (
            train_windows[
                fold_ml_feature_cols
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )


        # ----------------------------------------------------
        # TEST FEATURES
        # ----------------------------------------------------

        X_te = (
            test_windows[
                fold_ml_feature_cols
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )


        # ----------------------------------------------------
        # TARGETS
        # ----------------------------------------------------

        y_tr = le.transform(
            train_windows[
                "attack_label"
            ]
        )


        y_te = le.transform(
            test_windows[
                "attack_label"
            ]
        )


        # ----------------------------------------------------
        # FULL ML PREPROCESSING PIPELINE
        # ----------------------------------------------------
        #
        # SimpleImputer:
        #   fitted ONLY on X_tr
        #
        # StandardScaler:
        #   fitted ONLY on X_tr
        #
        # Classifier:
        #   fitted ONLY on X_tr
        #
        # X_te is transformed using parameters learned from
        # X_tr only.
        # ----------------------------------------------------

        pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "clf",
                clone(clf)
            )
        ])


        pipe.fit(
            X_tr,
            y_tr
        )


        preds = pipe.predict(
            X_te
        )


        # ----------------------------------------------------
        # OVERALL METRICS
        # ----------------------------------------------------

        accs.append(
            accuracy_score(
                y_te,
                preds
            )
        )


        macro_f1s.append(
            f1_score(
                y_te,
                preds,
                average="macro",
                labels=np.arange(
                    len(le.classes_)
                ),
                zero_division=0
            )
        )


        # ----------------------------------------------------
        # CLASS-WISE METRICS
        # ----------------------------------------------------

        p, r, f, s = (
            precision_recall_fscore_support(
                y_te,
                preds,
                labels=np.arange(
                    len(le.classes_)
                ),
                zero_division=0
            )
        )


        for idx, cls_name in enumerate(
            le.classes_
        ):

            precisions[
                cls_name
            ].append(
                p[idx]
            )


            recalls[
                cls_name
            ].append(
                r[idx]
            )


            f1s[
                cls_name
            ].append(
                f[idx]
            )


            supports[
                cls_name
            ].append(
                s[idx]
            )


        # ----------------------------------------------------
        # SAVE FIRST EVALUATED SPLIT
        # ----------------------------------------------------
        #
        # This split is used later for Figures 12 and 13.
        # It is processed by the same leakage-safe pipeline.
        #

        if not first_split_saved:

            first_split_outputs[
                m_name
            ] = {

                "model": pipe,

                "X_test": X_te,

                "y_test": y_te,

                "pred": preds,

                "feature_cols":
                    fold_ml_feature_cols,

                "train_sids":
                    sorted(
                        train_sids
                    ),

                "test_sids":
                    sorted(
                        test_sids
                    )
            }


            first_split_saved = True


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    eval_results[
        m_name
    ] = {

        "acc_mean":
            np.mean(accs),

        "acc_std":
            np.std(accs),

        "f1_mean":
            np.mean(macro_f1s),

        "f1_std":
            np.std(macro_f1s),

        "precision": {
            c: (
                np.mean(
                    precisions[c]
                ),
                np.std(
                    precisions[c]
                )
            )
            for c in le.classes_
        },

        "recalls": {
            c: (
                np.mean(
                    recalls[c]
                ),
                np.std(
                    recalls[c]
                )
            )
            for c in le.classes_
        },

        "f1s": {
            c: (
                np.mean(
                    f1s[c]
                ),
                np.std(
                    f1s[c]
                )
            )
            for c in le.classes_
        },

        "supports": {
            c: (
                np.mean(
                    supports[c]
                ),
                np.std(
                    supports[c]
                )
            )
            for c in le.classes_
        }
    }


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"\nModel: {m_name}"
    )


    print(
        f"Accuracy: "
        f"{eval_results[m_name]['acc_mean']:.4f}"
        f" ± "
        f"{eval_results[m_name]['acc_std']:.4f}"
    )


    print(
        f"Macro-F1: "
        f"{eval_results[m_name]['f1_mean']:.4f}"
        f" ± "
        f"{eval_results[m_name]['f1_std']:.4f}"
    )


    for c in le.classes_:

        print(
            f"  Precision ({c}): "
            f"{eval_results[m_name]['precision'][c][0]:.4f}"
            f" ± "
            f"{eval_results[m_name]['precision'][c][1]:.4f}"
        )


        print(
            f"  Recall ({c}): "
            f"{eval_results[m_name]['recalls'][c][0]:.4f}"
            f" ± "
            f"{eval_results[m_name]['recalls'][c][1]:.4f}"
        )


        print(
            f"  F1 ({c}): "
            f"{eval_results[m_name]['f1s'][c][0]:.4f}"
            f" ± "
            f"{eval_results[m_name]['f1s'][c][1]:.4f}"
        )


# ============================================================
# USE FIRST LEAKAGE-SAFE EXTRATREES SPLIT
# FOR FIGURES 12 AND 13
# ============================================================

if "ExtraTrees" not in first_split_outputs:

    raise RuntimeError(
        "ExtraTrees evaluation did not produce a split."
    )


first_et = (
    first_split_outputs[
        "ExtraTrees"
    ]
)


model = first_et[
    "model"
]


y_test = first_et[
    "y_test"
]


pred = first_et[
    "pred"
]


ml_feature_cols = first_et[
    "feature_cols"
]


# ============================================================
# Figure 12
# ============================================================

cm = confusion_matrix(
    y_test,
    pred,
    labels=np.arange(
        len(le.classes_)
    )
)


plt.figure(
    figsize=(7, 6)
)


plt.imshow(cm)


plt.title(
    "Figure 12. Leakage-safe attack impact classification"
)


plt.xlabel(
    "Predicted class"
)


plt.ylabel(
    "True class"
)


plt.xticks(
    range(len(le.classes_)),
    le.classes_,
    rotation=45,
    ha="right"
)


plt.yticks(
    range(len(le.classes_)),
    le.classes_
)


plt.colorbar()


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


save_numbered_figure(
    "leakage_safe_attack_impact_confusion_matrix"
)


# ============================================================
# Figure 13
# ============================================================

clf = model.named_steps[
    "clf"
]


importances = pd.DataFrame({

    "feature":
        ml_feature_cols,

    "importance":
        clf.feature_importances_

}).sort_values(
    "importance",
    ascending=False
).head(20)


importances.to_csv(
    f"{OUT}/feature_importance_top20.csv",
    index=False
)


plt.figure(
    figsize=(9, 7)
)


plt.barh(
    importances[
        "feature"
    ][::-1],

    importances[
        "importance"
    ][::-1]
)


plt.xlabel(
    "Importance"
)


plt.title(
    "Figure 13. Top window-level features for interference impact recognition"
)


save_numbered_figure(
    "top_window_features_interference_impact_recognition"
)


# ============================================================
# 8. SUMMARY FIGURE
# ============================================================

et_res = (
    eval_results[
        "ExtraTrees"
    ]
)


summary = pd.DataFrame({

    "Metric": [

        "Positioning samples",

        "Scenarios",

        "Attack classes",

        "Window samples",

        "Median radial drift [m]",

        "95th percentile radial drift [m]",

        "Median hAcc [m]",

        "95th percentile hAcc [m]",

        "Median integrity risk",

        "95th percentile integrity risk",

        "Repeated Group-CV Accuracy",

        "Repeated Group-CV Macro-F1"
    ],


    "Value": [

        len(data2),

        data2[
            "scenario_id"
        ].nunique(),

        ", ".join(
            sorted(
                data2[
                    "attack_label"
                ].unique()
            )
        ),

        len(windows_base),

        f"{np.median(data2['radial_drift_m']):.6f}",

        f"{np.percentile(data2['radial_drift_m'], 95):.6f}",

        f"{np.median(data2['hAcc']):.6f}",

        f"{np.percentile(data2['hAcc'], 95):.6f}",

        f"{np.median(data2['integrity_risk_index']):.4f}",

        f"{np.percentile(data2['integrity_risk_index'], 95):.4f}",

        f"{et_res['acc_mean']:.4f} ± {et_res['acc_std']:.4f}",

        f"{et_res['f1_mean']:.4f} ± {et_res['f1_std']:.4f}"
    ]
})


summary.to_csv(
    f"{OUT}/summary_metrics.csv",
    index=False
)


fig, ax = plt.subplots(
    figsize=(10, 4.8)
)


ax.axis(
    "off"
)


table = ax.table(
    cellText=summary.values,
    colLabels=summary.columns,
    cellLoc="left",
    loc="center"
)


table.auto_set_font_size(
    False
)


table.set_fontsize(
    10
)


table.scale(
    1,
    1.55
)


plt.title(
    f"Figure {fig_counter}. Summary of extracted GNSS positioning integrity data"
)


save_numbered_figure(
    "summary_of_extracted_positioning_integrity_data"
)


# ============================================================
# 9. CREATE PANELS OF 4 FIGURES
# ============================================================

def create_panel_of_4(
    paths,
    output_path,
    panel_title,
    cols=2,
    thumb_width=900,
    pad=45
):

    imgs = []


    for p in paths:

        img = (
            Image
            .open(p)
            .convert("RGB")
        )


        ratio = (
            thumb_width
            /
            img.width
        )


        img = img.resize(
            (
                thumb_width,
                int(
                    img.height
                    *
                    ratio
                )
            )
        )


        img = ImageOps.expand(
            img,
            border=2,
            fill="black"
        )


        imgs.append(
            (
                p,
                img
            )
        )


    if len(imgs) == 0:
        return


    rows = math.ceil(
        len(imgs)
        /
        cols
    )


    title_h = 110
    label_h = 60


    cell_w = (
        thumb_width
        +
        2 * pad
    )


    cell_h = (
        max(
            img.height
            for _, img in imgs
        )
        +
        label_h
        +
        2 * pad
    )


    canvas = Image.new(
        "RGB",
        (
            cols * cell_w,
            title_h + rows * cell_h
        ),
        "white"
    )


    draw = ImageDraw.Draw(
        canvas
    )


    try:

        font_title = ImageFont.truetype(
            "DejaVuSans-Bold.ttf",
            44
        )

        font_label = ImageFont.truetype(
            "DejaVuSans-Bold.ttf",
            28
        )

    except Exception:

        font_title = None
        font_label = None


    draw.text(
        (pad, 25),
        panel_title,
        fill="black",
        font=font_title
    )


    for i, (p, img) in enumerate(
        imgs
    ):

        r = i // cols
        c = i % cols


        x = (
            c * cell_w
            +
            pad
        )


        y = (
            title_h
            +
            r * cell_h
            +
            pad
            +
            label_h
        )


        fname = (
            os.path
            .basename(p)
            .replace(
                ".png",
                ""
            )
            .replace(
                "_",
                " "
            )
        )


        draw.text(
            (
                x,
                title_h
                +
                r * cell_h
                +
                pad
            ),
            fname,
            fill="black",
            font=font_label
        )


        canvas.paste(
            img,
            (
                x,
                y
            )
        )


    canvas.save(
        output_path,
        quality=95
    )


    print(
        "Saved panel:",
        output_path
    )


figure_registry = sorted(
    glob.glob(
        FIG_DIR
        +
        "/Figure_*.png"
    )
)


panel_paths = []


for i in range(
    0,
    len(figure_registry),
    4
):

    group = figure_registry[
        i:i + 4
    ]


    panel_no = (
        i // 4
        +
        1
    )


    out_panel = (
        f"{PANEL_DIR}/"
        f"Panel_{panel_no:02d}_"
        f"Figures_{i+1:02d}_to_"
        f"{i+len(group):02d}.png"
    )


    create_panel_of_4(
        group,
        out_panel,
        panel_title=(
            "GNSS Positioning Integrity Results "
            f"- Panel {panel_no}"
        ),
        cols=2,
        thumb_width=900
    )


    panel_paths.append(
        out_panel
    )


# ============================================================
# TRAJECTORY PANEL
# ============================================================

trajectory_panel_sources = []


for fig_no in [
    "Figure_02",
    "Figure_03",
    "Figure_04"
]:

    matches = sorted(
        glob.glob(
            FIG_DIR
            +
            f"/{fig_no}_*.png"
        )
    )


    if len(matches) > 0:

        trajectory_panel_sources.append(
            matches[0]
        )


        shutil.copy(
            matches[0],
            os.path.join(
                TRAJ_DIR,
                os.path.basename(
                    matches[0]
                )
            )
        )


trajectory_panel = (
    f"{TRAJ_DIR}/"
    "FINAL_PANEL_representative_trajectory_figures.png"
)


create_panel_of_4(
    trajectory_panel_sources,
    trajectory_panel,
    panel_title=(
        "Representative GNSS "
        "Trajectory Degradation Examples"
    ),
    cols=2,
    thumb_width=1000
)


# ============================================================
# 10. MASTER CONTACT SHEET
# ============================================================

def create_master_from_panels(
    panel_paths,
    output_path,
    thumb_width=1200,
    pad=35
):

    imgs = []


    for p in panel_paths:

        img = (
            Image
            .open(p)
            .convert("RGB")
        )


        ratio = (
            thumb_width
            /
            img.width
        )


        img = img.resize(
            (
                thumb_width,
                int(
                    img.height
                    *
                    ratio
                )
            )
        )


        imgs.append(
            img
        )


    if len(imgs) == 0:
        return


    total_h = (
        sum(
            img.height
            for img in imgs
        )
        +
        pad * (
            len(imgs)
            +
            1
        )
    )


    canvas = Image.new(
        "RGB",
        (
            thumb_width
            +
            2 * pad,
            total_h
        ),
        "white"
    )


    y = pad


    for img in imgs:

        canvas.paste(
            img,
            (
                pad,
                y
            )
        )


        y += (
            img.height
            +
            pad
        )


    canvas.save(
        output_path,
        quality=95
    )


    print(
        "Saved master contact sheet:",
        output_path
    )


master_path = (
    f"{OUT}/"
    "MASTER_All_Figure_Panels.png"
)


create_master_from_panels(
    panel_paths,
    master_path
)


# ============================================================
# 11. ZIP EVERYTHING
# ============================================================

zip_base = (
    "/content/"
    "GNSS_FINAL_Q1_POSITIONING_OUTPUTS"
)


if os.path.exists(
    zip_base + ".zip"
):

    os.remove(
        zip_base + ".zip"
    )


shutil.make_archive(
    zip_base,
    "zip",
    OUT
)


print(
    "\nDONE."
)


print(
    "Separate numbered figures:",
    FIG_DIR
)


print(
    "Trajectory figures:",
    TRAJ_DIR
)


print(
    "Panels of 4 figures:",
    PANEL_DIR
)


print(
    "Trajectory panel:",
    trajectory_panel
)


print(
    "Master contact sheet:",
    master_path
)


print(
    "ZIP file:",
    zip_base + ".zip"
)