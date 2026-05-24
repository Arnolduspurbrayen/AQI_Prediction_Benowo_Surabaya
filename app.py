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
            hparams_best.json
            tft_best_state_dict.pt
        AQI_CN/
            tft_Benowo_aqi_cn_best.ckpt
            hparams_best.json
            tft_best_state_dict.pt
    pure_test_15_USpct.csv
    pure_test_15_CNpct.csv
    requirements.txt
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import warnings
warnings.filterwarnings("ignore")

import unittest.mock as mock
from pathlib import Path
from datetime import timedelta, datetime

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go

# ── Path config ────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent

# Model configs per target
MODEL_CONFIGS = {
    "AQI US": {
        "target"    : "aqi_us",
        "label"     : "AQI US",
        "ckpt_path" : APP_DIR / "models" / "AQI_US" / "tft_Benowo_aqi_us_best.ckpt",
        "csv_path"  : APP_DIR / "data" /"pure_test_15_USpct.csv",
        "color"     : "#636EFA",
        "color2"    : "rgba(99,110,250,0.15)",
    },
    "AQI CN": {
        "target"    : "aqi_cn",
        "label"     : "AQI CN",
        "ckpt_path" : APP_DIR / "models" / "AQI_CN" / "tft_Benowo_aqi_cn_best.ckpt",
        "csv_path"  : APP_DIR / "data" / "pure_test_15_CNpct.csv",
        "color"     : "#00CC96",
        "color2"    : "rgba(0,204,150,0.15)",
    },
}

# ── TFT config (shared) ───────────────────────────────────────────────────────
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
    layout="centered",
)

# ── Mobile-first minimal UI ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:      #0a0d12;
  --card:    #111620;
  --card2:   #181f2e;
  --border:  rgba(255,255,255,0.07);
  --text:    #e2e8f0;
  --muted:   #64748b;
  --accent:  #38bdf8;
  --green:   #34d399;
  --red:     #f87171;
  --radius:  14px;
  --font:    'Inter', sans-serif;
  --mono:    'JetBrains Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}

/* ── HIDE CHROME ── */
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stSidebar"]        { display: none !important; }

/* ── CONTAINER ── */
.block-container {
    padding: 1rem 1rem 3rem !important;
    max-width: 680px !important;
    margin: 0 auto !important;
}
@media (min-width: 768px) {
    .block-container {
        padding: 1.5rem 1.5rem 3rem !important;
        max-width: 860px !important;
    }
}

/* ── HERO ── */
.hero {
    padding: 1.25rem 0 0.5rem;
}
.hero-title {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
    line-height: 1.3;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 0.78rem !important;
    color: var(--muted) !important;
}
.hero-badge {
    display: inline-block;
    background: rgba(52,211,153,0.12);
    color: var(--green);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.68rem;
    font-family: var(--mono);
    font-weight: 600;
    letter-spacing: 0.05em;
    vertical-align: middle;
    margin-left: 0.5rem;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 0.8rem 0 !important;
}

/* ── MODEL SELECTOR (radio) ── */
[data-testid="stRadio"] > label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
    margin-bottom: 0.4rem !important;
}
[data-baseweb="radio"] { gap: 0.5rem !important; }
[data-baseweb="radio"] label {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.82rem !important;
    color: var(--text) !important;
    transition: border-color 0.15s !important;
}
[data-baseweb="radio"] label:has(input:checked) {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(56,189,248,0.06) !important;
}

/* ── METRIC CARDS ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.9rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.6rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── AQI CATEGORY BADGE (Rule 8 fix: visible on mobile) ── */
.aqi-badge-inline {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    font-family: var(--mono);
    border-radius: 5px;
    padding: 0.15rem 0.5rem;
    margin-top: 0.3rem;
    letter-spacing: 0.04em;
}

/* ── COLUMNS gap ── */
[data-testid="stHorizontalBlock"] { gap: 0.6rem !important; }

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 2px !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    padding: 0.45rem 0.85rem !important;
    border: none !important;
    white-space: nowrap !important;
    transition: all 0.15s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--card2) !important;
    color: var(--text) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabsContent"] { padding-top: 1rem !important; }

/* ── CHART WRAPPER ── */
[data-testid="stPlotlyChart"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    background: var(--card) !important;
}

/* ── SECTION LABEL ── */
.section-label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
    margin-bottom: 0.6rem !important;
    margin-top: 1rem !important;
    display: block;
}

/* ── SLIDER ── */
[data-testid="stSlider"] [role="progressbar"] {
    background: linear-gradient(90deg, var(--accent), var(--green)) !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumb"] {
    background: var(--accent) !important;
    width: 20px !important; height: 20px !important;
}
[data-testid="stSlider"] > label {
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}

/* ── INPUTS ── */
[data-baseweb="input"] > div,
[data-baseweb="select"] > div:first-child {
    background: var(--card2) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
input, textarea { color: var(--text) !important; }
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}

/* ── BUTTONS ── */
button[kind="primary"] {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
}
button[kind="primary"]:hover { opacity: 0.85 !important; }
button[kind="secondary"], button:not([kind]) {
    background: var(--card2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
}
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    border-radius: 10px !important;
    font-size: 0.8rem !important;
    width: 100% !important;
}

/* ── PRESET BUTTONS (Rule 2 fix) ── */
.preset-row {
    display: flex;
    gap: 0.4rem;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
}
.preset-label {
    font-size: 0.63rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 0.35rem;
    display: block;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    padding: 0.8rem 1rem !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
    font-size: 0.78rem !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
    font-size: 0.82rem !important;
}

/* ── PROGRESS ── */
[data-testid="stProgress"] > div {
    background: linear-gradient(90deg, var(--accent), var(--green)) !important;
    border-radius: 6px !important;
}
[data-testid="stProgress"] {
    background: var(--card2) !important;
    border-radius: 6px !important;
}

/* ── TYPOGRAPHY ── */
h2, h3 {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.25rem !important;
}
.stMarkdown p {
    font-size: 0.82rem !important;
    color: var(--muted) !important;
    line-height: 1.6 !important;
}
strong { color: var(--text) !important; }
code {
    background: rgba(56,189,248,0.1) !important;
    color: var(--accent) !important;
    border-radius: 5px !important;
    font-family: var(--mono) !important;
    font-size: 0.78em !important;
    padding: 0.1em 0.35em !important;
}


/* ── INFO CARDS ── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin: 0.9rem 0 0.5rem;
}
@media (max-width: 500px) { .info-grid { grid-template-columns: 1fr; } }
.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem 0.9rem;
}
.info-card-label {
    font-size: 0.63rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #475569;
    margin-bottom: 0.25rem;
}
.info-card-val {
    font-size: 0.84rem;
    font-weight: 500;
    color: #cbd5e1;
    line-height: 1.5;
}
.info-card-val small {
    display: block;
    font-size: 0.72rem;
    color: #475569;
    margin-top: 0.1rem;
}

/* ── AQI LEGEND ── */
.aqi-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin: 0.4rem 0 0.8rem;
}
.aqi-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #111620;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px;
    padding: 0.22rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: #94a3b8;
    font-family: var(--mono);
}
.aqi-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── GUIDE BOX ── */
.guide-box {
    background: #111620;
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid #38bdf8;
    border-radius: var(--radius);
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
}
.guide-box p {
    font-size: 0.8rem !important;
    color: #94a3b8 !important;
    line-height: 1.7 !important;
    margin: 0 !important;
}
.guide-box b { color: #cbd5e1 !important; }
.guide-box ol {
    margin: 0.4rem 0 0 1.1rem;
    padding: 0;
}
.guide-box li {
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.7;
    padding-left: 0.2rem;
}
.guide-box li b { color: #cbd5e1; }

/* ── SECTION DIVIDER LABEL ── */
.ctx-label {
    font-size: 0.63rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin: 0.6rem 0 0.3rem;
    display: block;
}

/* ── METRIC HINT ── */
.mhint {
    font-size: 0.75rem;
    color: #64748b;
    line-height: 1.6;
    margin: 0.25rem 0 0.75rem;
}
.mhint b { color: #94a3b8; }

/* ── ERROR BOX (Rule 5 fix: user-friendly) ── */
.error-box {
    background: rgba(248,81,73,0.08);
    border: 1px solid rgba(248,81,73,0.25);
    border-left: 3px solid #f85149;
    border-radius: var(--radius);
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
}
.error-box-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #f87171;
    margin-bottom: 0.3rem;
}
.error-box-msg {
    font-size: 0.78rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--card2); border-radius: 2px; }

/* ── MOBILE STACKS ── */
@media (max-width: 600px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.5rem !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: none !important;
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    .block-container { padding: 0.75rem 0.75rem 3rem !important; }
    button[kind="primary"] { padding: 0.65rem 1rem !important; }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        font-size: 0.72rem !important;
        padding: 0.4rem 0.6rem !important;
    }
}
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
    if val <= 50:  return "Excellent",            "#00e400"
    if val <= 100: return "Good",                 "#ffff00"
    if val <= 150: return "Lightly Polluted",     "#ff7e00"
    if val <= 200: return "Moderately Polluted",  "#ff0000"
    if val <= 300: return "Heavily Polluted",     "#8f3f97"
    return "Severely Polluted",                   "#7e0023"


def aqi_category(val, target="aqi_us"):
    if target == "aqi_cn":
        return aqi_cn_category(val)
    return aqi_us_category(val)


# ── Rule 8 fix: render badge HTML visible on mobile ──────────────────────────
def aqi_badge_html(val, target):
    label, color = aqi_category(val, target)
    # pick readable text color
    text_color = "#000" if color in ("#00e400", "#ffff00", "#ff7e00") else "#fff"
    return (
        f'<span class="aqi-badge-inline" '
        f'style="background:{color}22;color:{color};border:1px solid {color}55;">'
        f'{label}</span>'
    )


def get_unknown_reals(target: str):
    return [target, "pm25", "pm10",
            f"{target}_lag1", f"{target}_lag24", f"{target}_lag168",
            "pm25_lag1", "pm25_lag24",
            "pm10_lag1", "pm10_lag24"]


def add_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
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
# ERROR HELPERS — Rule 5: user-friendly messages
# ══════════════════════════════════════════════════════════════════════════════

def show_friendly_error(title: str, message: str, detail: str = None):
    """Tampilkan error yang ramah user tanpa stack trace teknis."""
    html = f"""
    <div class="error-box">
      <div class="error-box-title">⚠️ {title}</div>
      <div class="error-box-msg">{message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    if detail:
        with st.expander("Detail teknis (untuk developer)"):
            st.code(detail, language="text")


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


# ══════════════════════════════════════════════════════════════════════════════
# AQI REFERENCE LINES
# ══════════════════════════════════════════════════════════════════════════════

AQI_HLINES_US = [
    (50,  "Good",          "#39d353"),
    (100, "Moderate",      "#d29922"),
    (150, "Sensitive",     "#ff7e00"),
    (200, "Unhealthy",     "#f85149"),
    (300, "V. Unhealthy",  "#8f3f97"),
]

AQI_HLINES_CN = [
    (50,  "Excellent",         "#39d353"),
    (100, "Good",              "#d29922"),
    (150, "Lightly Polluted",  "#ff7e00"),
    (200, "Mod. Polluted",     "#f85149"),
    (300, "Heavily Polluted",  "#8f3f97"),
]

def add_aqi_hlines(fig, target="aqi_us"):
    lines = AQI_HLINES_CN if target == "aqi_cn" else AQI_HLINES_US
    for level, label, color in lines:
        fig.add_hline(
            y=level, line_dash="dot", line_color=color,
            annotation_text=label,
            annotation_position="bottom right",
            annotation_font_size=10,
        )
    return fig


def style_fig(fig, mobile_height=300):
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
        title_font=dict(family="Inter, sans-serif", size=13, color="#cbd5e1", weight=600),
        legend=dict(
            bgcolor="rgba(17,22,32,0.9)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(size=10, color="#94a3b8"),
            orientation="h",
            yanchor="top", y=-0.15,
            xanchor="left", x=0,
        ),
        margin=dict(l=8, r=8, t=56, b=8),
        hoverlabel=dict(
            bgcolor="#1c2333", bordercolor="rgba(255,255,255,0.1)",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#e6edf3"),
        ),
        dragmode="zoom",
    )
    return fig


PLOTLY_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "displaylogo": False,
    "responsive": True,
}


# ══════════════════════════════════════════════════════════════════════════════
# HEADER + MODEL SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="hero-title">AQI Forecast &middot; Benowo</div>
  <div class="hero-sub">Prediksi kualitas udara 24 jam ke depan &middot; Stasiun Benowo, Surabaya</div>
</div>

<span class="ctx-label">Tentang aplikasi ini</span>
<div class="info-grid">
  <div class="info-card">
    <div class="info-card-label">Stasiun</div>
    <div class="info-card-val">Benowo, Surabaya
      <small>Data per jam (hourly)</small>
    </div>
  </div>
  <div class="info-card">
    <div class="info-card-label">Model Prediksi</div>
    <div class="info-card-val">Temporal Fusion Transformer
      <small>Prediksi H+1 hingga H+24</small>
    </div>
  </div>
  <div class="info-card">
    <div class="info-card-label">Standar AQI</div>
    <div class="info-card-val">
      <span style="color:#636EFA;font-weight:600;">AQI US</span> <span style="color:#475569;">—</span> EPA Amerika
      <small><span style="color:#00CC96;font-weight:600;">AQI CN</span> <span style="color:#475569;">—</span> China GB3095</small>
    </div>
  </div>
  <div class="info-card">
    <div class="info-card-label">Data yang Digunakan</div>
    <div class="info-card-val">Data uji (15% akhir)
      <small>Belum pernah dilihat model</small>
    </div>
  </div>
</div>

<span class="ctx-label">Perbandingan Skala AQI · Referensi IQAir</span>
<div style="overflow-x:auto;margin-bottom:0.5rem;border-radius:10px;border:1px solid rgba(255,255,255,0.07);overflow:hidden;">
<table style="width:100%;border-collapse:collapse;font-size:0.72rem;font-family:'JetBrains Mono',monospace;min-width:520px;">
  <thead>
    <tr style="background:#0f1520;border-bottom:1px solid rgba(255,255,255,0.1);">
      <th style="text-align:left;padding:0.55rem 0.75rem;color:#636EFA;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;width:26%;">🇺🇸 US AQI Level</th>
      <th style="text-align:center;padding:0.55rem 0.5rem;color:#64748b;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.06em;width:22%;">US PM2.5<br><span style="font-size:0.55rem;">(µg/m³)</span></th>
      <th style="text-align:center;padding:0.55rem 0.5rem;color:#64748b;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.06em;width:22%;">China PM2.5<br><span style="font-size:0.55rem;">(µg/m³)</span></th>
      <th style="text-align:right;padding:0.55rem 0.75rem;color:#00CC96;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;width:30%;">🇨🇳 China AQI Level</th>
    </tr>
    <tr style="background:#0a0f1a;border-bottom:1px solid rgba(255,255,255,0.07);">
      <td colspan="4" style="padding:0.28rem 0.75rem;font-size:0.6rem;color:#334155;font-family:'Inter',sans-serif;text-align:center;letter-spacing:0.03em;">
        WHO PM2.5 Recommended Guidelines 2024 &nbsp;·&nbsp; <b style="color:#475569;">0 – 5.0 µg/m³</b>
      </td>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.04);background:rgba(0,228,0,0.04);">
      <td style="padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#00e400;font-weight:700;">Good</span>
        <span style="color:#334155;font-size:0.65rem;margin-left:0.3rem;">0–50</span>
      </td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">0 – 9.0</td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">0 – 35</td>
      <td style="text-align:right;padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#00e400;font-weight:700;">Excellent</span>
        <span style="color:#334155;font-size:0.65rem;margin-left:0.3rem;">0–50</span>
      </td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.04);background:rgba(180,160,0,0.04);">
      <td style="padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#c8b800;font-weight:700;">Moderate</span>
        <span style="color:#334155;font-size:0.65rem;margin-left:0.3rem;">51–100</span>
      </td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">9.1 – 35.4</td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">35.1 – 75</td>
      <td style="text-align:right;padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#c8b800;font-weight:700;">Good</span>
        <span style="color:#334155;font-size:0.65rem;margin-left:0.3rem;">51–100</span>
      </td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.04);background:rgba(255,126,0,0.04);">
      <td style="padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#ff7e00;font-weight:700;">Unhealthy for<br>Sensitive Groups</span>
        <span style="color:#334155;font-size:0.65rem;display:block;margin-top:0.1rem;">101–150</span>
      </td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">35.5 – 55.4</td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">75.1 – 115</td>
      <td style="text-align:right;padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#ff7e00;font-weight:700;">Lightly<br>Polluted</span>
        <span style="color:#334155;font-size:0.65rem;display:block;margin-top:0.1rem;">101–150</span>
      </td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.04);background:rgba(220,50,50,0.04);">
      <td style="padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#e05252;font-weight:700;">Unhealthy</span>
        <span style="color:#334155;font-size:0.65rem;margin-left:0.3rem;">151–200</span>
      </td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">55.5 – 125.4</td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">115.1 – 150</td>
      <td style="text-align:right;padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#e05252;font-weight:700;">Moderately<br>Polluted</span>
        <span style="color:#334155;font-size:0.65rem;display:block;margin-top:0.1rem;">151–200</span>
      </td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.04);background:rgba(143,63,151,0.04);">
      <td style="padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#9b59b6;font-weight:700;">Very Unhealthy</span>
        <span style="color:#334155;font-size:0.65rem;margin-left:0.3rem;">201–300</span>
      </td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">125.5 – 225.4</td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">150.1 – 250</td>
      <td style="text-align:right;padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#9b59b6;font-weight:700;">Heavily<br>Polluted</span>
        <span style="color:#334155;font-size:0.65rem;display:block;margin-top:0.1rem;">201–300</span>
      </td>
    </tr>
    <tr style="background:rgba(126,0,35,0.04);">
      <td style="padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#b03030;font-weight:700;">Hazardous</span>
        <span style="color:#334155;font-size:0.65rem;margin-left:0.3rem;">301+</span>
      </td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">225.5+</td>
      <td style="text-align:center;padding:0.5rem 0.5rem;color:#64748b;vertical-align:middle;">250.1 – 500</td>
      <td style="text-align:right;padding:0.5rem 0.75rem;vertical-align:middle;">
        <span style="color:#b03030;font-weight:700;">Severely<br>Polluted</span>
        <span style="color:#334155;font-size:0.65rem;display:block;margin-top:0.1rem;">301+</span>
      </td>
    </tr>
  </tbody>
</table>
</div>
<p style="font-size:0.65rem;color:#334155;margin-bottom:0.7rem;font-family:'Inter',sans-serif;line-height:1.6;">
  Sumber: <b style="color:#475569;">IQAir · EPA 2024 · China GB3095</b> &nbsp;·&nbsp; Rentang AQI kedua standar sama (0–500), ambang PM2.5 CN lebih longgar.
</p>
""", unsafe_allow_html=True)

st.markdown('<span class="ctx-label">Pilih standar AQI yang digunakan</span>', unsafe_allow_html=True)
selected_model = st.radio(
    "Pilih standar AQI",
    list(MODEL_CONFIGS.keys()),
    index=0,
    horizontal=True,
    help="AQI US menggunakan standar EPA Amerika. AQI CN menggunakan standar China GB3095. Keduanya memakai skala 0–500.",
    label_visibility="collapsed",
)

cfg        = MODEL_CONFIGS[selected_model]
TARGET     = cfg["target"]
CKPT_PATH  = cfg["ckpt_path"]
TEST_CSV   = cfg["csv_path"]
LINE_COLOR = cfg["color"]
FILL_COLOR = cfg["color2"]

# ── Rule 5: Friendly file-not-found error ────────────────────────────────────
if not CKPT_PATH.exists():
    show_friendly_error(
        "File Model Tidak Ditemukan",
        f"Model <code>{selected_model}</code> belum tersedia di folder <code>models/</code>. "
        "Pastikan file checkpoint sudah ditempatkan dengan benar sesuai struktur folder.",
        detail=str(CKPT_PATH),
    )
    st.stop()

if not TEST_CSV.exists():
    show_friendly_error(
        "File Data Tidak Ditemukan",
        f"File data uji untuk <code>{selected_model}</code> tidak ditemukan. "
        "Pastikan file CSV sudah ada di folder <code>data/</code>.",
        detail=str(TEST_CSV),
    )
    st.stop()

df_b = load_test_data(str(TEST_CSV), TARGET)

with st.spinner(f"Memuat model {selected_model}..."):
    try:
        model, training_dataset, n_test = load_model_and_dataset(
            str(CKPT_PATH), df_b, TARGET)
    except Exception as e:
        show_friendly_error(
            "Gagal Memuat Model",
            "Terjadi masalah saat memuat model prediksi. "
            "Coba refresh halaman. Jika masalah berlanjut, periksa versi library di <code>requirements.txt</code>.",
            detail=str(e),
        )
        st.stop()

n_test_eff = int(len(df_b) * 0.15)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, = st.tabs(["Prediksi"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f'<span class="section-label">Prediksi {selected_model} · Data Test</span>', unsafe_allow_html=True)
    st.markdown("""
<div class="guide-box">
  <p><b>Cara menggunakan halaman ini</b></p>
  <ol>
    <li>Pilih titik awal prediksi menggunakan <b>tombol preset</b> (Awal / Tengah / Akhir) atau geser <b>slider</b> secara manual.</li>
    <li>Klik <b>Reset</b> kapan saja untuk kembali ke posisi awal.</li>
    <li>Model akan memprediksi <b>24 jam ke depan</b> secara otomatis.</li>
    <li>Grafik menampilkan prediksi (garis biru), nilai aktual (garis merah putus), dan rentang ketidakpastian Q10&ndash;Q90 (area abu).</li>
    <li>Kartu di atas grafik menunjukkan nilai & kategori AQI jam pertama (H+1) serta error keseluruhan 24 jam.</li>
  </ol>
</div>
""", unsafe_allow_html=True)

    test_start_row = len(df_b) - n_test_eff
    max_offset     = max(0, n_test_eff - MAX_PREDICTION_LENGTH - 1)

    if max_offset <= 0:
        show_friendly_error(
            "Data Tidak Cukup",
            "Jumlah data uji terlalu sedikit untuk menjalankan prediksi 24 jam. "
            "Pastikan dataset memiliki minimal 192 baris data per stasiun.",
        )
        st.stop()

    # ── Rule 2: Preset shortcuts + Rule 6: Reset button ──────────────────────
    st.markdown('<span class="preset-label">Pilih cepat titik prediksi</span>', unsafe_allow_html=True)

    # Inisialisasi session state untuk slider
    if "slider_offset" not in st.session_state:
        st.session_state["slider_offset"] = 0

    col_awal, col_tengah, col_akhir, col_reset = st.columns([1, 1, 1, 1])
    with col_awal:
        if st.button("⏮ Awal", use_container_width=True, help="Lompat ke awal periode data uji"):
            st.session_state["slider_offset"] = 0
    with col_tengah:
        if st.button("⏭ Tengah", use_container_width=True, help="Lompat ke pertengahan periode data uji"):
            st.session_state["slider_offset"] = (max_offset // 2 // MAX_PREDICTION_LENGTH) * MAX_PREDICTION_LENGTH
    with col_akhir:
        if st.button("⏩ Akhir", use_container_width=True, help="Lompat ke akhir periode data uji"):
            st.session_state["slider_offset"] = (max_offset // MAX_PREDICTION_LENGTH) * MAX_PREDICTION_LENGTH
    with col_reset:
        if st.button("↺ Reset", use_container_width=True, help="Kembalikan ke posisi awal (jam ke-0)"):
            st.session_state["slider_offset"] = 0

    offset = st.slider(
        "Geser titik awal prediksi (jam ke-N dalam periode test)",
        min_value=0,
        max_value=max_offset,
        value=st.session_state["slider_offset"],
        step=MAX_PREDICTION_LENGTH,
        key="slider_offset",
        help="0 = awal data test. Gunakan tombol preset di atas untuk lompat cepat.",
    )

    pred_start_row = test_start_row + offset
    window_start   = max(0, pred_start_row - MAX_ENCODER_LENGTH)
    window_end     = min(len(df_b), pred_start_row + MAX_PREDICTION_LENGTH)
    df_window      = df_b.iloc[window_start:window_end].copy()
    pred_dt_start  = df_b["datetime_final"].iloc[pred_start_row]

    st.caption(
        f"Periode: {pred_dt_start.strftime('%d %b %Y %H:00')} → "
        f"{(pred_dt_start + timedelta(hours=MAX_PREDICTION_LENGTH-1)).strftime('%d %b %Y %H:00')}"
    )

    with st.spinner("Menjalankan prediksi…"):
        try:
            pred_q, actual = run_prediction(model, training_dataset, df_window)
        except Exception as e:
            show_friendly_error(
                "Prediksi Gagal",
                "Model tidak dapat menjalankan prediksi untuk periode ini. "
                "Coba geser slider ke posisi lain, atau gunakan tombol <b>Reset</b> untuk kembali ke awal.",
                detail=str(e),
            )
            st.stop()

    q10, q50, q90 = pred_q[:, 0], pred_q[:, 1], pred_q[:, 2]
    dt_index = [pred_dt_start + timedelta(hours=i) for i in range(MAX_PREDICTION_LENGTH)]

    mae_val      = float(np.mean(np.abs(q50 - actual)))
    rmse_val     = float(np.sqrt(np.mean((q50 - actual) ** 2)))
    cat_pred, _  = aqi_category(q50[0], TARGET)
    cat_act,  _  = aqi_category(actual[0], TARGET)

    # ── Rule 8: Tampilkan kategori langsung di bawah nilai, visible di mobile ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{selected_model} Prediksi (H+1)", f"{q50[0]:.1f}")
    c2.metric(f"{selected_model} Aktual (H+1)",   f"{actual[0]:.1f}")
    c3.metric("MAE (24 jam)",  f"{mae_val:.2f}",  help="Rata-rata selisih absolut antara prediksi dan nilai aktual selama 24 jam")
    c4.metric("RMSE (24 jam)", f"{rmse_val:.2f}", help="Seperti MAE, namun error besar diberi penalti lebih tinggi")

    # Badge kategori AQI di bawah metric — selalu terlihat, termasuk mobile
    badge_pred = aqi_badge_html(q50[0], TARGET)
    badge_act  = aqi_badge_html(actual[0], TARGET)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(badge_pred, unsafe_allow_html=True)
    with b2:
        st.markdown(badge_act, unsafe_allow_html=True)
    with b3:
        st.markdown(
            '<span style="font-size:0.68rem;color:#475569;font-family:var(--mono)">rata-rata error</span>',
            unsafe_allow_html=True
        )
    with b4:
        st.markdown(
            '<span style="font-size:0.68rem;color:#475569;font-family:var(--mono)">error ± penalti</span>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="mhint"><b>H+1</b> = jam pertama dari window prediksi. '
        'Kategori AQI ditampilkan langsung di bawah nilai. '
        'MAE dan RMSE dihitung dari seluruh 24 jam prediksi.</div>',
        unsafe_allow_html=True
    )

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
    fig = add_aqi_hlines(fig, TARGET)
    fig.update_layout(
        title=f"Prediksi 24 Jam · {pred_dt_start.strftime('%d %b %Y')}",
        xaxis_title="Waktu", yaxis_title=selected_model,
        hovermode="x unified", height=360, template="plotly_dark",
    )
    fig = style_fig(fig)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

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
        st.download_button("⬇️ Download CSV", csv_dl,
                           f"prediksi_{TARGET}_{pred_dt_start.strftime('%Y%m%d_%H%M')}.csv",
                           "text/csv")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:0.5rem 0 1rem;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:#94a3b8;">
    AQI Prediction &middot; Benowo, Surabaya &middot; TFT
  </span>
  <br>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:#64748b;">
    Binus University &middot; Jakarta &middot; 2026
  </span>
</div>
""", unsafe_allow_html=True)
