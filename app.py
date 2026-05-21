"""
TFT AQI Forecasting App — Stasiun Benowo
Streamlit app untuk prediksi AQI US & AQI CN 24 jam ke depan menggunakan Temporal Fusion Transformer.

Cara menjalankan:
    streamlit run app.py

Struktur folder:
    app.py
    models/
        AQI_US/
            tft_Benowo_aqi_us_best.ckpt
        AQI_CN/
            tft_Benowo_aqi_cn_best.ckpt
    data/
        pure_test_15_USpct.csv
        pure_test_15_CNpct.csv
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import warnings
warnings.filterwarnings("ignore")

import unittest.mock as mock
from pathlib import Path
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go

# ── Path config ────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent

MODEL_CONFIGS = {
    "AQI US": {
        "target"   : "aqi_us",
        "label"    : "AQI US",
        "ckpt_path": APP_DIR / "models" / "AQI_US" / "tft_Benowo_aqi_us_best.ckpt",
        "csv_path" : APP_DIR / "data" / "pure_test_15_USpct.csv",
        "color"    : "#636EFA",
        "color2"   : "rgba(99,110,250,0.15)",
    },
    "AQI CN": {
        "target"   : "aqi_cn",
        "label"    : "AQI CN",
        "ckpt_path": APP_DIR / "models" / "AQI_CN" / "tft_Benowo_aqi_cn_best.ckpt",
        "csv_path" : APP_DIR / "data" / "pure_test_15_CNpct.csv",
        "color"    : "#00CC96",
        "color2"   : "rgba(0,204,150,0.15)",
    },
}

STATION               = "Benowo"
MAX_ENCODER_LENGTH    = 192
MIN_ENCODER_LENGTH    = 96
MAX_PREDICTION_LENGTH = 24
QUANTILES             = [0.1, 0.5, 0.9]

TIME_VARYING_KNOWN_CAT  = ["month_cat", "hour_cat", "dayofweek_cat", "is_weekend_cat"]
TIME_VARYING_KNOWN_REAL = ["time_idx", "year", "hour_sin", "hour_cos",
                            "dow_sin", "dow_cos", "month_sin", "month_cos"]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Forecast · Benowo",
    page_icon="🌬️",
    layout="wide",
)

# ── Responsive CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #0d1117;
  --surface:   #161b22;
  --surface2:  #1c2333;
  --border:    rgba(255,255,255,0.08);
  --border2:   rgba(255,255,255,0.05);
  --text:      #e6edf3;
  --muted:     #7d8590;
  --accent:    #39d353;
  --accent2:   #58a6ff;
  --warn:      #d29922;
  --danger:    #f85149;
  --purple:    #bc8cff;
  --radius:    14px;
  --radius-sm: 9px;
  --shadow:    0 4px 24px rgba(0,0,0,0.4);
  --font:      'Inter', sans-serif;
  --font-head: 'Syne', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

html, body,
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 60% 40% at 10% 10%, rgba(57,211,83,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 90% 80%, rgba(88,166,255,0.05) 0%, transparent 60%);
}

#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; }

/* ── MAIN CONTAINER (fluid, mobile-first) ── */
.block-container {
    padding: 0.75rem 0.75rem 4rem !important;
    max-width: 100% !important;
    width: 100% !important;
}
@media (min-width: 600px) {
    .block-container { padding: 1.25rem 1.5rem 4rem !important; }
}
@media (min-width: 1024px) {
    .block-container { padding: 1.75rem 2.5rem 4rem !important; }
}
@media (min-width: 1400px) {
    .block-container {
        padding: 2rem 3.5rem 4rem !important;
        max-width: 1500px !important;
        margin: 0 auto !important;
    }
}

/* ── HERO ── */
.aqi-hero {
    display: flex; flex-wrap: wrap; align-items: center;
    gap: 0.6rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    position: relative; overflow: hidden;
}
.aqi-hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--purple));
}
.aqi-hero-icon {
    width: 40px; height: 40px; flex-shrink: 0;
    background: linear-gradient(135deg, rgba(57,211,83,0.15), rgba(88,166,255,0.15));
    border: 1px solid rgba(57,211,83,0.25);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
}
.aqi-hero-body { flex: 1; min-width: 160px; }
.aqi-hero-body h1 {
    font-family: var(--font-head) !important;
    font-size: clamp(0.9rem, 2.8vw, 1.25rem) !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
    margin: 0 0 2px !important;
}
.aqi-hero-body p {
    font-size: clamp(0.68rem, 1.6vw, 0.78rem) !important;
    color: var(--muted) !important;
    margin: 0 !important;
}
.aqi-pill {
    background: rgba(57,211,83,0.1);
    border: 1px solid rgba(57,211,83,0.25);
    color: var(--accent);
    border-radius: 30px; padding: 0.25rem 0.75rem;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.04em; white-space: nowrap;
    font-family: var(--font-mono);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 1rem 0.9rem !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--font-head) !important;
    font-size: 0.68rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    color: var(--muted) !important; margin-bottom: 0.5rem !important;
}
[data-testid="stSidebar"] .stRadio > label {
    font-size: 0.82rem !important; color: var(--text) !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }
[data-testid="stSidebar"] .stMarkdown p { font-size: 0.78rem !important; color: var(--muted) !important; }
[data-testid="stSidebar"] code {
    background: rgba(88,166,255,0.1) !important; color: var(--accent2) !important;
    border-radius: 4px !important; font-family: var(--font-mono) !important;
    font-size: 0.72rem !important; padding: 0.1em 0.3em !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button { width: 100% !important; }

/* ── METRICS — fluid grid ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.85rem 1rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
    width: 100%;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(57,211,83,0.3) !important;
    box-shadow: 0 4px 20px rgba(57,211,83,0.1) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.09em !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: clamp(1.2rem, 3vw, 1.85rem) !important;
    font-weight: 500 !important; color: var(--text) !important;
    line-height: 1.15 !important;
}

/* ── COLUMNS: stack on small screens ── */
/* Let Streamlit's default column flex work; only stack at very small */
@media (max-width: 480px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.5rem !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: none !important;
    }
}
@media (min-width: 481px) and (max-width: 767px) {
    /* 2-column on medium-small: let flex wrap */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }
    [data-testid="column"] {
        min-width: calc(50% - 0.5rem) !important;
        flex: 1 1 calc(50% - 0.5rem) !important;
    }
}
[data-testid="stHorizontalBlock"] { gap: 0.65rem !important; }

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important; gap: 3px !important;
    flex-wrap: wrap !important;
    overflow-x: auto !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: var(--radius-sm) !important;
    font-size: clamp(0.7rem, 1.8vw, 0.82rem) !important;
    font-weight: 600 !important; color: var(--muted) !important;
    padding: 0.45rem 0.85rem !important; border: none !important;
    transition: all 0.18s !important; white-space: nowrap;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--surface2) !important; color: var(--text) !important;
    box-shadow: 0 0 0 1px var(--border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabsContent"] { padding-top: 0.9rem !important; }

/* ── BUTTONS ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #2ea043, #1a7f37) !important;
    color: #fff !important; border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important; font-size: 0.86rem !important;
    padding: 0.5rem 1.2rem !important;
    box-shadow: 0 0 16px rgba(46,160,67,0.3) !important;
    transition: all 0.18s !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 0 24px rgba(46,160,67,0.5) !important;
    transform: translateY(-1px) !important;
}
button[kind="secondary"], button:not([kind]) {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.83rem !important;
    transition: border-color 0.18s !important;
}
button[kind="secondary"]:hover, button:not([kind]):hover {
    border-color: var(--accent2) !important;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] button {
    background: rgba(88,166,255,0.08) !important;
    color: var(--accent2) !important;
    border: 1px solid rgba(88,166,255,0.25) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
}
[data-testid="stDownloadButton"] button:hover { background: rgba(88,166,255,0.15) !important; }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important; font-size: 0.83rem !important;
    color: var(--text) !important; padding: 0.8rem 1rem !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    width: 100% !important;
}
[data-testid="stDataFrame"] table {
    background: var(--surface) !important;
    font-size: 0.78rem !important;
    font-family: var(--font-mono) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    border: none !important; font-size: 0.83rem !important;
}

/* ── INPUTS ── */
[data-baseweb="input"], [data-baseweb="select"] > div:first-child {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
}
input, textarea, select {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border-radius: var(--radius-sm) !important;
}
/* number input width on mobile */
[data-testid="stNumberInput"] { width: 100% !important; }

/* ── SLIDER ── */
[data-testid="stSlider"] [role="progressbar"] {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumb"] {
    background: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(57,211,83,0.25) !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    background: var(--surface2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important;
    font-family: var(--font-mono) !important; font-size: 0.72rem !important;
    border-radius: 5px !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrack"] { background: var(--surface2) !important; }

/* ── PROGRESS ── */
[data-testid="stProgress"] > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 6px !important;
}
[data-testid="stProgress"] { background: var(--surface2) !important; border-radius: 6px !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] svg { color: var(--accent) !important; }

/* ── TYPOGRAPHY ── */
h2 {
    font-family: var(--font-head) !important;
    font-size: clamp(0.9rem, 2.2vw, 1.1rem) !important;
    font-weight: 700 !important; color: var(--text) !important;
    letter-spacing: -0.01em !important;
}
h3 {
    font-family: var(--font-head) !important;
    font-size: clamp(0.82rem, 1.9vw, 0.97rem) !important;
    font-weight: 600 !important; color: var(--text) !important;
}
.stMarkdown p {
    font-size: clamp(0.78rem, 1.8vw, 0.86rem) !important;
    color: var(--muted) !important; line-height: 1.65 !important;
}
strong { color: var(--text) !important; }
code {
    background: rgba(88,166,255,0.1) !important; color: var(--accent2) !important;
    border-radius: 4px !important; font-family: var(--font-mono) !important;
    font-size: 0.8em !important; padding: 0.1em 0.3em !important;
}
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1rem 0 !important; }
.stCaption, [data-testid="stCaptionContainer"] {
    font-family: var(--font-mono) !important; font-size: 0.68rem !important; color: var(--muted) !important;
}

/* ── POPOVER / SELECT ── */
[data-baseweb="popover"] { background: var(--surface2) !important; }
[data-baseweb="menu"] { background: var(--surface2) !important; }
[data-baseweb="menu"] li { color: var(--text) !important; }
[data-baseweb="menu"] li:hover { background: var(--surface) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def aqi_us_category(val):
    if val is None or np.isnan(float(val)):
        return "N/A", "#888888"
    val = float(val)
    if val <= 50:  return "Good",                    "#00e400"
    if val <= 100: return "Moderate",                "#ffff00"
    if val <= 150: return "Unhealthy for Sensitive", "#ff7e00"
    if val <= 200: return "Unhealthy",               "#ff0000"
    if val <= 300: return "Very Unhealthy",          "#8f3f97"
    return "Hazardous",                              "#7e0023"


def aqi_cn_category(val):
    if val is None or np.isnan(float(val)):
        return "N/A", "#888888"
    val = float(val)
    if val <= 50:  return "Excellent", "#00e400"
    if val <= 100: return "Good",      "#ffff00"
    if val <= 150: return "Light",     "#ff7e00"
    if val <= 200: return "Moderate",  "#ff0000"
    if val <= 300: return "Heavy",     "#8f3f97"
    return "Severe",                   "#7e0023"


def aqi_category(val, target="aqi_us"):
    if target == "aqi_cn":
        return aqi_cn_category(val)
    return aqi_us_category(val)


def get_unknown_reals(target: str):
    return [target, "pm25", "pm10",
            f"{target}_lag1", f"{target}_lag24", f"{target}_lag168",
            "pm25_lag1", "pm25_lag24",
            "pm10_lag1", "pm10_lag24"]


def add_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Tambahkan semua fitur engineered yang sama dengan saat training."""
    df = df.copy()
    df["month_cat"]      = df["month"].astype(str)
    df["hour_cat"]       = df["hour"].astype(str)
    df["dayofweek_cat"]  = df["dayofweek"].astype(str)
    df["is_weekend_cat"] = (df["dayofweek"] >= 5).astype(int).astype(str)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]      / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]      / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]     / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]     / 12)
    for col, lags in [(target, [1, 24, 168]), ("pm25", [1, 24]), ("pm10", [1, 24])]:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag).bfill()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Memuat data…")
def load_test_data(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime_final"] = pd.to_datetime(df["datetime_final"])
    df["station"]        = df["station"].astype(str)
    df = df.sort_values(["station", "datetime_final"]).reset_index(drop=True)
    df = df[df["station"] == STATION].copy().reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df = add_features(df, target)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Memuat model TFT…")
def load_model_and_dataset(ckpt_path: str, df: pd.DataFrame, target: str):
    with mock.patch("torch.cuda.is_available", return_value=False):
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp   = ckpt["hyper_parameters"]
        sd   = ckpt["state_dict"]

        model = TemporalFusionTransformer(**hp)
        model.load_state_dict(sd, strict=False)
        model.eval()

        n_test         = int(len(df) * 0.15)
        dataset_params = ckpt.get("dataset_parameters", None)

        if dataset_params is not None:
            training_dataset = TimeSeriesDataSet.from_parameters(
                dataset_params, df,
                predict=False, stop_randomization=True,
            )
        else:
            df_train = df.iloc[:-n_test].copy() if n_test > 0 else df.copy()
            training_dataset = TimeSeriesDataSet(
                df_train,
                time_idx                        = "time_idx",
                target                          = target,
                group_ids                       = ["station"],
                min_encoder_length              = MIN_ENCODER_LENGTH,
                max_encoder_length              = MAX_ENCODER_LENGTH,
                min_prediction_length           = MAX_PREDICTION_LENGTH,
                max_prediction_length           = MAX_PREDICTION_LENGTH,
                static_categoricals             = ["station"],
                time_varying_known_categoricals = TIME_VARYING_KNOWN_CAT,
                time_varying_known_reals        = TIME_VARYING_KNOWN_REAL,
                time_varying_unknown_reals      = get_unknown_reals(target),
                target_normalizer               = GroupNormalizer(
                                                    groups=["station"], center=False),
                add_relative_time_idx  = True,
                add_target_scales      = True,
                add_encoder_length     = True,
                allow_missing_timesteps= True,
            )

        return model, training_dataset, n_test


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_prediction(model, training_dataset, df_window: pd.DataFrame):
    """
    Jalankan inferensi TFT pada df_window.
    df_window harus sudah punya time_idx yang kontinu dan semua kolom fitur.
    """
    with mock.patch("torch.cuda.is_available", return_value=False):
        from pytorch_forecasting import TimeSeriesDataSet

        min_pred_idx = int(df_window["time_idx"].max()) - MAX_PREDICTION_LENGTH + 1
        pred_ds = TimeSeriesDataSet.from_dataset(
            training_dataset, df_window,
            predict=True, stop_randomization=True,
            min_prediction_idx=min_pred_idx,
        )
        loader = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
        batch  = next(iter(loader))
        x, y   = batch

        with torch.no_grad():
            out = model(x)

        pred_q = out["prediction"].numpy()[0]
        actual = (y[0] if isinstance(y, tuple) else y).numpy()[0]
        return pred_q[:MAX_PREDICTION_LENGTH], actual[:MAX_PREDICTION_LENGTH]


def build_manual_window(df_base: pd.DataFrame, target: str,
                         manual_aqi: float, manual_pm25: float, manual_pm10: float) -> pd.DataFrame:
    """
    FIX: Bangun window context untuk input manual dengan benar.

    Langkah:
    1. Ambil MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH baris terakhir dari df_base
    2. Override nilai sensor pada baris TERAKHIR (jam paling baru)
    3. Hitung ulang lag features SETELAH override agar konsisten
    4. Reset time_idx supaya kontinu dan tidak konflik dengan training_dataset
    """
    # Ambil window yang cukup panjang
    n_need  = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH
    df_ctx  = df_base.tail(n_need).copy().reset_index(drop=True)

    # Override nilai sensor pada baris terakhir (jam "sekarang")
    last_idx = df_ctx.index[-1]
    df_ctx.at[last_idx, target]  = manual_aqi
    df_ctx.at[last_idx, "pm25"] = manual_pm25
    df_ctx.at[last_idx, "pm10"] = manual_pm10

    # Hitung ulang lag features agar konsisten dengan override di atas
    for col, lags in [(target, [1, 24, 168]), ("pm25", [1, 24]), ("pm10", [1, 24])]:
        for lag in lags:
            df_ctx[f"{col}_lag{lag}"] = df_ctx[col].shift(lag).bfill()

    # Pastikan time_idx kontinu (dimulai dari nilai terakhir di df_base minus panjang window)
    # Kita gunakan time_idx asli yang sudah ada, cukup reset agar bersih
    base_time_idx = int(df_base["time_idx"].iloc[-(n_need)])
    df_ctx["time_idx"] = range(base_time_idx, base_time_idx + len(df_ctx))

    # Pastikan kolom station ada dan bertipe string
    df_ctx["station"] = STATION

    return df_ctx


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

AQI_HLINES = [
    (50,  "Good",      "#39d353"),
    (100, "Moderate",  "#d29922"),
    (150, "Sensitive", "#ff7e00"),
    (200, "Unhealthy", "#f85149"),
]

def add_aqi_hlines(fig):
    for level, label, color in AQI_HLINES:
        fig.add_hline(
            y=level, line_dash="dot", line_color=color,
            annotation_text=label,
            annotation_position="bottom right",
            annotation_font_size=10,
        )
    return fig


def style_fig(fig):
    BG   = "#161b22"
    GRID = "rgba(255,255,255,0.05)"
    TICK = "#7d8590"
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1,
        showline=False, zeroline=False,
        tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TICK),
        title_font=dict(family="Inter, sans-serif", size=11, color=TICK),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1,
        showline=False, zeroline=False,
        tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TICK),
        title_font=dict(family="Inter, sans-serif", size=11, color=TICK),
    )
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color="#e6edf3"),
        title_font=dict(family="Syne, sans-serif", size=14, color="#e6edf3"),
        legend=dict(
            bgcolor="rgba(22,27,34,0.8)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(size=11, color="#e6edf3"),
        ),
        margin=dict(l=8, r=8, t=44, b=8),
        hoverlabel=dict(
            bgcolor="#1c2333", bordercolor="rgba(255,255,255,0.1)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#e6edf3"),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="aqi-hero">
  <div class="aqi-hero-icon">🌬️</div>
  <div class="aqi-hero-body">
    <h1>AQI Forecast · Stasiun Benowo</h1>
    <p>Prediksi kualitas udara 24 jam ke depan · Temporal Fusion Transformer · EXP3</p>
  </div>
  <div class="aqi-pill">● LIVE</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Konfigurasi")

selected_model = st.sidebar.radio(
    "🎯 Pilih Target Model:",
    list(MODEL_CONFIGS.keys()),
    index=0,
    help="AQI US = standar EPA Amerika. AQI CN = standar China (GB3095).",
)

cfg        = MODEL_CONFIGS[selected_model]
TARGET     = cfg["target"]
CKPT_PATH  = cfg["ckpt_path"]
TEST_CSV   = cfg["csv_path"]
LINE_COLOR = cfg["color"]
FILL_COLOR = cfg["color2"]

st.sidebar.markdown(f"Model aktif: **`{selected_model}`** (`{TARGET}`)")
st.sidebar.markdown("---")

if not CKPT_PATH.exists():
    st.error(f"❌ Checkpoint tidak ditemukan: `{CKPT_PATH}`")
    st.info(f"Pastikan file ada di: `models/{selected_model.replace(' ', '_')}/`")
    st.stop()
if not TEST_CSV.exists():
    st.error(f"❌ Data tidak ditemukan: `{TEST_CSV}`")
    st.stop()

df_b = load_test_data(str(TEST_CSV), TARGET)

with st.spinner(f"Memuat model TFT ({selected_model})…"):
    try:
        model, training_dataset, n_test = load_model_and_dataset(
            str(CKPT_PATH), df_b, TARGET)
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

n_test_eff = int(len(df_b) * 0.15)
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Info Data ({selected_model}):**
- Total     : `{len(df_b):,}` jam
- Periode   : `{df_b['datetime_final'].iloc[0].date()}` → `{df_b['datetime_final'].iloc[-1].date()}`
- Test (15%): `{n_test_eff}` jam terakhir
""")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊 Prediksi dari Data Test",
    "✏️ Input Manual",
    "📈 Evaluasi Keseluruhan",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Prediksi dari Data Test
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Prediksi {selected_model} dari Data Test (15% terakhir)")
    st.markdown("Geser slider untuk memilih titik awal prediksi dalam periode test.")

    test_start_row = len(df_b) - n_test_eff
    max_offset     = max(0, n_test_eff - MAX_PREDICTION_LENGTH - 1)

    if max_offset <= 0:
        st.warning("Data test terlalu sedikit untuk prediksi. Tambah jumlah data.")
        st.stop()

    offset = st.slider(
        "Geser titik awal prediksi (jam ke-N dalam periode test)",
        min_value=0, max_value=max_offset, value=0, step=MAX_PREDICTION_LENGTH,
        help="0 = awal data test",
    )

    pred_start_row = test_start_row + offset
    window_start   = max(0, pred_start_row - MAX_ENCODER_LENGTH)
    window_end     = min(len(df_b), pred_start_row + MAX_PREDICTION_LENGTH)
    df_window      = df_b.iloc[window_start:window_end].copy()
    pred_dt_start  = df_b["datetime_final"].iloc[pred_start_row]

    st.markdown(
        f"**Periode prediksi:** "
        f"`{pred_dt_start.strftime('%Y-%m-%d %H:00')}` → "
        f"`{(pred_dt_start + timedelta(hours=MAX_PREDICTION_LENGTH-1)).strftime('%Y-%m-%d %H:00')}`"
    )

    with st.spinner("Menjalankan prediksi…"):
        try:
            pred_q, actual = run_prediction(model, training_dataset, df_window)
        except Exception as e:
            st.error(f"❌ Prediksi gagal: {e}")
            st.exception(e)
            st.stop()

    q10, q50, q90 = pred_q[:, 0], pred_q[:, 1], pred_q[:, 2]
    dt_index = [pred_dt_start + timedelta(hours=i) for i in range(MAX_PREDICTION_LENGTH)]

    mae_val      = float(np.mean(np.abs(q50 - actual)))
    rmse_val     = float(np.sqrt(np.mean((q50 - actual) ** 2)))
    cat_pred, _  = aqi_category(q50[0], TARGET)
    cat_act,  _  = aqi_category(actual[0], TARGET)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{selected_model} Prediksi (H+1)", f"{q50[0]:.1f}",    help=cat_pred)
    c2.metric(f"{selected_model} Aktual (H+1)",   f"{actual[0]:.1f}", help=cat_act)
    c3.metric("MAE (24 jam)",  f"{mae_val:.2f}")
    c4.metric("RMSE (24 jam)", f"{rmse_val:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dt_index + dt_index[::-1],
        y=list(q90) + list(q10[::-1]),
        fill="toself", fillcolor=FILL_COLOR,
        line=dict(color="rgba(255,255,255,0)"),
        name="Interval Q10–Q90", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dt_index, y=q50,
        mode="lines+markers", line=dict(color=LINE_COLOR, width=2),
        marker=dict(size=5), name="Prediksi (Q50)",
    ))
    fig.add_trace(go.Scatter(
        x=dt_index, y=actual,
        mode="lines+markers", line=dict(color="#f85149", width=2, dash="dash"),
        marker=dict(size=5), name="Aktual",
    ))
    fig = add_aqi_hlines(fig)
    fig.update_layout(
        title=f"Prediksi {selected_model} 24 Jam — Mulai {pred_dt_start.strftime('%d %b %Y %H:00')}",
        xaxis_title="Waktu", yaxis_title=selected_model,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=420, template="plotly_dark",
    )
    fig = style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Tabel Detail Prediksi vs Aktual"):
        df_result = pd.DataFrame({
            "Waktu"                : [t.strftime("%Y-%m-%d %H:%M") for t in dt_index],
            "Q10"                  : np.round(q10, 2),
            "Q50 (Prediksi)"       : np.round(q50, 2),
            "Q90"                  : np.round(q90, 2),
            "Aktual"               : np.round(actual, 2),
            "Error |Q50 - Aktual|" : np.round(np.abs(q50 - actual), 2),
            "Kategori Prediksi"    : [aqi_category(v, TARGET)[0] for v in q50],
            "Kategori Aktual"      : [aqi_category(v, TARGET)[0] for v in actual],
        })
        st.dataframe(df_result, use_container_width=True, hide_index=True)
        csv_dl = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV", csv_dl,
            f"prediksi_{TARGET}_{pred_dt_start.strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Input Manual  (BUG FIXED)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"✏️ Input Manual Nilai {selected_model} Terkini")
    st.markdown(
        "Masukkan nilai sensor terbaru. Model akan menggunakan **192 jam data historis** "
        "dari dataset, lalu **jam terakhir diganti** dengan nilai yang kamu masukkan, "
        "kemudian memprediksi **24 jam berikutnya**."
    )

    col_a, col_b, col_c = st.columns(3)
    manual_aqi  = col_a.number_input(
        f"{selected_model} saat ini", min_value=0.0, max_value=500.0, value=100.0, step=1.0,
        key="manual_aqi",
    )
    manual_pm25 = col_b.number_input(
        "PM2.5 saat ini", min_value=0.0, max_value=500.0, value=50.0, step=1.0,
        key="manual_pm25",
    )
    manual_pm10 = col_c.number_input(
        "PM10 saat ini", min_value=0.0, max_value=500.0, value=80.0, step=1.0,
        key="manual_pm10",
    )

    if st.button("🚀 Jalankan Prediksi Manual", type="primary", key="btn_manual"):
        with st.spinner("Membangun context window dan menjalankan prediksi…"):
            try:
                # ── FIX: gunakan helper yang benar ──────────────────────────
                df_ctx = build_manual_window(
                    df_b, TARGET, manual_aqi, manual_pm25, manual_pm10
                )

                # Prediksi: tidak ada "actual" yang bermakna di sini
                pred_q2, _ = run_prediction(model, training_dataset, df_ctx)
                q10b, q50b, q90b = pred_q2[:, 0], pred_q2[:, 1], pred_q2[:, 2]

                # Timestamp prediksi dimulai dari +1 jam setelah data terakhir
                last_dt   = df_b["datetime_final"].iloc[-1]
                start_dt2 = last_dt + timedelta(hours=1)
                dt_idx2   = [start_dt2 + timedelta(hours=i) for i in range(MAX_PREDICTION_LENGTH)]

                # ── Metrics ──
                ca, cb, cc = st.columns(3)
                cat_h1, _  = aqi_category(q50b[0],  TARGET)
                cat_h12, _ = aqi_category(q50b[11], TARGET)
                cat_h24, _ = aqi_category(q50b[-1], TARGET)
                ca.metric(f"{selected_model} H+1",  f"{q50b[0]:.1f}",  help=cat_h1)
                cb.metric(f"{selected_model} H+12", f"{q50b[11]:.1f}", help=cat_h12)
                cc.metric(f"{selected_model} H+24", f"{q50b[-1]:.1f}", help=cat_h24)

                st.info(f"Kategori {selected_model} jam +1: **{cat_h1}**")

                # ── Chart ──
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=dt_idx2 + dt_idx2[::-1],
                    y=list(q90b) + list(q10b[::-1]),
                    fill="toself", fillcolor=FILL_COLOR,
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Interval Q10–Q90", hoverinfo="skip",
                ))
                fig2.add_trace(go.Scatter(
                    x=dt_idx2, y=q50b,
                    mode="lines+markers", line=dict(color=LINE_COLOR, width=2),
                    marker=dict(size=5), name="Prediksi (Q50)",
                ))
                fig2 = add_aqi_hlines(fig2)
                fig2.update_layout(
                    title=f"Prediksi {selected_model} 24 Jam ke Depan (Input Manual)",
                    xaxis_title="Waktu", yaxis_title=selected_model,
                    height=420, template="plotly_dark", hovermode="x unified",
                )
                fig2 = style_fig(fig2)
                st.plotly_chart(fig2, use_container_width=True)

                with st.expander("📋 Lihat semua nilai prediksi"):
                    df_manual_res = pd.DataFrame({
                        "Waktu"         : [t.strftime("%Y-%m-%d %H:%M") for t in dt_idx2],
                        "Q10"           : np.round(q10b, 2),
                        "Q50 (Prediksi)": np.round(q50b, 2),
                        "Q90"           : np.round(q90b, 2),
                        "Kategori"      : [aqi_category(v, TARGET)[0] for v in q50b],
                    })
                    st.dataframe(df_manual_res, use_container_width=True, hide_index=True)
                    csv_m = df_manual_res.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download CSV", csv_m,
                        f"prediksi_manual_{TARGET}.csv", "text/csv",
                        key="dl_manual",
                    )

            except Exception as e:
                st.error(f"❌ Prediksi gagal: {e}")
                st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Evaluasi Keseluruhan
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"📈 Evaluasi Model {selected_model} pada Data Test")
    st.markdown(
        "Sampling prediksi setiap **N jam** pada periode test. "
        "Klik tombol untuk memulai (butuh beberapa menit)."
    )

    sample_step = st.selectbox(
        "Interval sampling (jam)",
        [24, 48, 72, 168], index=1,
        help="24 = setiap hari, 48 = setiap 2 hari, 168 = setiap 7 hari",
        key="sample_step",
    )

    if st.button("▶️ Jalankan Evaluasi", type="primary", key="btn_eval"):
        test_start_row = len(df_b) - n_test_eff
        windows = list(range(0, n_test_eff - MAX_PREDICTION_LENGTH, sample_step))

        prog   = st.progress(0)
        status = st.empty()
        all_q50, all_actual, all_dt = [], [], []

        for i, off in enumerate(windows):
            prog.progress((i + 1) / max(len(windows), 1))
            status.text(f"Memproses window {i+1}/{len(windows)}…")

            ps  = test_start_row + off
            ws  = max(0, ps - MAX_ENCODER_LENGTH)
            we  = min(len(df_b), ps + MAX_PREDICTION_LENGTH)
            dfw = df_b.iloc[ws:we].copy()

            try:
                pq, act = run_prediction(model, training_dataset, dfw)
                dt0     = df_b["datetime_final"].iloc[ps]
                all_q50.append(pq[:, 1])
                all_actual.append(act)
                all_dt.extend([dt0 + timedelta(hours=h) for h in range(MAX_PREDICTION_LENGTH)])
            except Exception:
                continue

        prog.empty()
        status.empty()

        if not all_q50:
            st.error("Tidak ada prediksi yang berhasil.")
        else:
            q50_all  = np.concatenate(all_q50)
            dt_all   = all_dt
            act_all  = np.concatenate(all_actual)
            mae_all  = float(np.mean(np.abs(q50_all - act_all)))
            rmse_all = float(np.sqrt(np.mean((q50_all - act_all) ** 2)))
            mape_all = float(np.mean(
                np.abs((q50_all - act_all) / np.clip(act_all, 1, None))) * 100)
            corr     = float(np.corrcoef(q50_all, act_all)[0, 1])

            st.success(f"✅ {len(windows)} window · {len(q50_all)} jam total")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE",          f"{mae_all:.2f}")
            m2.metric("RMSE",         f"{rmse_all:.2f}")
            m3.metric("MAPE",         f"{mape_all:.1f}%")
            m4.metric("Korelasi (r)", f"{corr:.3f}")

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=dt_all, y=q50_all,
                mode="lines", line=dict(color=LINE_COLOR, width=1.5),
                name="Prediksi (Q50)", opacity=0.85,
            ))
            fig3.add_trace(go.Scatter(
                x=dt_all, y=act_all,
                mode="lines", line=dict(color="#f85149", width=1.5, dash="dash"),
                name="Aktual", opacity=0.85,
            ))
            fig3 = add_aqi_hlines(fig3)
            fig3.update_layout(
                title=f"Prediksi vs Aktual {selected_model} — Periode Test",
                xaxis_title="Waktu", yaxis_title=selected_model,
                height=420, template="plotly_dark", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig3 = style_fig(fig3)
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=act_all, y=q50_all, mode="markers",
                marker=dict(color=q50_all, colorscale="Viridis",
                            size=4, opacity=0.5, showscale=True,
                            colorbar=dict(title=selected_model)),
                name="Prediksi vs Aktual",
            ))
            vmin = float(min(act_all.min(), q50_all.min()))
            vmax = float(max(act_all.max(), q50_all.max()))
            fig4.add_trace(go.Scatter(
                x=[vmin, vmax], y=[vmin, vmax],
                mode="lines", line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1),
                name="Ideal (y=x)",
            ))
            fig4.update_layout(
                title=f"Scatter: Aktual vs Prediksi ({selected_model})",
                xaxis_title=f"Aktual {selected_model}",
                yaxis_title=f"Prediksi {selected_model} (Q50)",
                height=420, template="plotly_dark",
            )
            fig4 = style_fig(fig4)
            st.plotly_chart(fig4, use_container_width=True)

            if len(all_q50) > 1:
                errs  = np.array([np.abs(pq - act)
                                  for pq, act in zip(all_q50, all_actual)])
                mae_h = errs.mean(axis=0)
                fig5  = go.Figure()
                fig5.add_trace(go.Bar(
                    x=list(range(1, MAX_PREDICTION_LENGTH + 1)),
                    y=mae_h,
                    marker_color="rgba(88,166,255,0.7)",
                    name="MAE per Horizon",
                ))
                fig5.update_layout(
                    title=f"MAE per Horizon (H+1 s/d H+{MAX_PREDICTION_LENGTH}) — {selected_model}",
                    xaxis_title="Horizon (jam)", yaxis_title="MAE",
                    height=340, template="plotly_dark",
                )
                fig5 = style_fig(fig5)
                st.plotly_chart(fig5, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:0.4rem 0 1rem;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#7d8590;">
    🌬️ AQI Forecast · Stasiun Benowo · TFT (pytorch-forecasting) · EXP3 · dropout=0.25 · lr=3e-05 · hidden=64 · lstm_layers=2
  </span>
</div>
""", unsafe_allow_html=True)
