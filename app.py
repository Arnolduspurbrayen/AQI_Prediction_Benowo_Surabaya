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
    margin: 0.8rem 0 0.6rem;
}
@media (max-width: 500px) { .info-grid { grid-template-columns: 1fr; } }
.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.7rem 0.85rem;
}
.info-card-icon { font-size: 1rem; margin-bottom: 0.1rem; }
.info-card-title {
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.15rem;
}
.info-card-val { font-size: 0.81rem; font-weight: 500; color: var(--text); line-height: 1.45; }
.info-card-val small { font-size: 0.71rem; color: var(--muted); }

/* ── AQI LEGEND CHIPS ── */
.aqi-legend { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.35rem 0 0.75rem; }
.aqi-chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    border-radius: 20px; padding: 0.18rem 0.55rem;
    font-size: 0.68rem; font-weight: 600; font-family: var(--mono);
    border: 1px solid rgba(255,255,255,0.08); background: var(--card); color: var(--text);
}
.aqi-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

/* ── GUIDE BOX ── */
.guide-box {
    background: rgba(56,189,248,0.04);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: var(--radius);
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.85rem;
}
.guide-title {
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; color: var(--accent); margin-bottom: 0.4rem;
}
.guide-step {
    font-size: 0.79rem; color: #94a3b8; line-height: 1.65;
    display: flex; align-items: flex-start; gap: 0.45rem; margin-bottom: 0.1rem;
}
.snum {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 1.3em; height: 1.3em;
    background: rgba(56,189,248,0.13); color: var(--accent);
    border-radius: 50%; font-size: 0.68rem; font-weight: 700; flex-shrink: 0; margin-top: 0.1em;
}

/* ── METRIC HINT ── */
.mhint {
    font-size: 0.73rem; color: var(--muted); margin: 0.2rem 0 0.7rem; line-height: 1.5;
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
    """Kategori AQI berdasarkan standar China (GB3095-2012)."""
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


def style_fig(fig, mobile_height=300):
    """Dark elegant chart styling with mobile config."""
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
    )
    # Config untuk mobile: allow pan & zoom tanpa block scroll
    fig.update_layout(
        dragmode="zoom",
    )
    return fig


# Config untuk semua chart: scroll mobile tidak terganggu
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
  <div class="hero-title">AQI Forecast &middot; Benowo <span class="hero-badge">&#9679; TFT</span></div>
  <div class="hero-sub">Prediksi kualitas udara 24 jam ke depan &middot; Stasiun Benowo, Surabaya &middot; Temporal Fusion Transformer</div>
</div>
<div class="info-grid">
  <div class="info-card">
    <div class="info-card-icon">🏭</div>
    <div class="info-card-title">Stasiun Pemantauan</div>
    <div class="info-card-val">Benowo, Surabaya<br><small>Data historis per jam (hourly)</small></div>
  </div>
  <div class="info-card">
    <div class="info-card-icon">🤖</div>
    <div class="info-card-title">Model AI</div>
    <div class="info-card-val">Temporal Fusion Transformer<br><small>Prediksi H+1 hingga H+24</small></div>
  </div>
  <div class="info-card">
    <div class="info-card-icon">🌏</div>
    <div class="info-card-title">Dua Standar AQI</div>
    <div class="info-card-val"><b>AQI US</b> — EPA Amerika<br><b>AQI CN</b> — China GB3095</div>
  </div>
  <div class="info-card">
    <div class="info-card-icon">📐</div>
    <div class="info-card-title">Data Uji</div>
    <div class="info-card-val">15% akhir dataset<br><small>Model belum pernah melihat data ini</small></div>
  </div>
</div>
<div class="section-label" style="margin-top:0;">Skala Kategori AQI (US &amp; CN)</div>
<div class="aqi-legend">
  <span class="aqi-chip"><span class="aqi-dot" style="background:#00e400"></span>0–50 Baik</span>
  <span class="aqi-chip"><span class="aqi-dot" style="background:#ffff00"></span>51–100 Sedang</span>
  <span class="aqi-chip"><span class="aqi-dot" style="background:#ff7e00"></span>101–150 Sensitif</span>
  <span class="aqi-chip"><span class="aqi-dot" style="background:#ff0000"></span>151–200 Tdk Sehat</span>
  <span class="aqi-chip"><span class="aqi-dot" style="background:#8f3f97"></span>201–300 Sgt Tdk Sehat</span>
  <span class="aqi-chip"><span class="aqi-dot" style="background:#7e0023"></span>&gt;300 Berbahaya</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label" style="margin-top:0.5rem;margin-bottom:0.3rem;">Pilih Standar AQI</div>', unsafe_allow_html=True)
selected_model = st.radio(
    "Pilih Standar AQI",
    list(MODEL_CONFIGS.keys()),
    index=0,
    horizontal=True,
    help="AQI US = standar EPA Amerika (skala 0-500). AQI CN = standar China GB3095 (skala 0-500). Pilih sesuai referensi yang ingin digunakan.",
    label_visibility="collapsed",
)

cfg        = MODEL_CONFIGS[selected_model]
TARGET     = cfg["target"]
CKPT_PATH  = cfg["ckpt_path"]
TEST_CSV   = cfg["csv_path"]
LINE_COLOR = cfg["color"]
FILL_COLOR = cfg["color2"]

if not CKPT_PATH.exists():
    st.error(f"Checkpoint tidak ditemukan: {CKPT_PATH}")
    st.stop()
if not TEST_CSV.exists():
    st.error(f"Data tidak ditemukan: {TEST_CSV}")
    st.stop()
df_b = load_test_data(str(TEST_CSV), TARGET)

with st.spinner(f"Memuat model {selected_model}..."):
    try:
        model, training_dataset, n_test = load_model_and_dataset(
            str(CKPT_PATH), df_b, TARGET)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

n_test_eff = int(len(df_b) * 0.15)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab3 = st.tabs(["📊 Prediksi", "📈 Evaluasi"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f'<span class="section-label">Prediksi {selected_model} · Data Test</span>', unsafe_allow_html=True)
    st.markdown("""
<div class="guide-box">
  <div class="guide-title">Cara Pakai Tab Prediksi</div>
  <div class="guide-step"><span class="snum">1</span>Geser <b>slider</b> untuk memilih titik awal prediksi di dalam periode data test.</div>
  <div class="guide-step"><span class="snum">2</span>Model otomatis memprediksi <b>24 jam ke depan</b> dari titik tersebut.</div>
  <div class="guide-step"><span class="snum">3</span>Grafik: <span style="color:#636EFA">&#9644; biru</span> = prediksi Q50, <span style="color:#f85149">- - merah</span> = aktual, area abu = ketidakpastian Q10&ndash;Q90.</div>
  <div class="guide-step"><span class="snum">4</span>Kartu metrik menampilkan nilai jam pertama (H+1) dan error rata-rata 24 jam (MAE &amp; RMSE).</div>
</div>
""", unsafe_allow_html=True)

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

    st.caption(
        f"Periode: {pred_dt_start.strftime('%d %b %Y %H:00')} → "
        f"{(pred_dt_start + timedelta(hours=MAX_PREDICTION_LENGTH-1)).strftime('%d %b %Y %H:00')}"
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
    c3.metric("MAE (24 jam)",  f"{mae_val:.2f}",  help="Mean Absolute Error — rata-rata selisih absolut prediksi vs aktual")
    c4.metric("RMSE (24 jam)", f"{rmse_val:.2f}", help="Root Mean Squared Error — error besar diberi penalti lebih tinggi")
    st.markdown(
        f'<div class="mhint">&#128161; <b>H+1</b> = jam pertama dari window prediksi. '
        f'Hover kartu untuk kategori AQI. MAE &amp; RMSE dihitung atas seluruh 24 jam.</div>',
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
    fig = add_aqi_hlines(fig)
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

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(f'<span class="section-label">Evaluasi · {selected_model}</span>', unsafe_allow_html=True)
    st.markdown("""
<div class="guide-box">
  <div class="guide-title">Cara Pakai Tab Evaluasi</div>
  <div class="guide-step"><span class="snum">1</span>Pilih <b>interval sampling</b> — seberapa sering model dijalankan. Misal: 24 = setiap hari, 168 = setiap minggu.</div>
  <div class="guide-step"><span class="snum">2</span>Klik <b>Jalankan Evaluasi</b>. Proses membutuhkan beberapa menit (makin kecil interval, makin lama).</div>
  <div class="guide-step"><span class="snum">3</span>Hasil: metrik akurasi global (MAE, RMSE, MAPE, Korelasi) + grafik prediksi vs aktual seluruh periode test.</div>
  <div class="guide-step"><span class="snum">4</span><b>MAE per Horizon</b> menunjukkan akurasi tiap jam ke depan — wajar bila error naik semakin jauh horizonnya.</div>
</div>
""", unsafe_allow_html=True)
    st.caption("Interval lebih kecil = lebih banyak window = evaluasi lebih menyeluruh, namun butuh waktu lebih lama.")

    sample_step = st.selectbox(
        "Interval sampling (jam)",
        [24, 48, 72, 168], index=1,
        help="24 = setiap hari, 48 = setiap 2 hari, 168 = setiap 7 hari",
    )

    if st.button("▶️ Jalankan Evaluasi", type="primary"):
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
            q50_all = np.concatenate(all_q50)
            dt_all  = all_dt

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
                title="Prediksi vs Aktual",
                xaxis_title="Waktu", yaxis_title=selected_model,
                height=360, template="plotly_dark", hovermode="x unified",
                    )
            fig3 = style_fig(fig3)
            st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)

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
                title="Scatter · Aktual vs Prediksi",
                xaxis_title=f"Aktual {selected_model}",
                yaxis_title=f"Prediksi {selected_model} (Q50)",
                height=360, template="plotly_dark",
            )
            fig4 = style_fig(fig4)
            st.plotly_chart(fig4, use_container_width=True, config=PLOTLY_CONFIG)

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
                    title="MAE per Horizon",
                    xaxis_title="Horizon (jam)", yaxis_title="MAE",
                    height=300, template="plotly_dark",
                )
                fig5 = style_fig(fig5)
                st.plotly_chart(fig5, use_container_width=True, config=PLOTLY_CONFIG)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:0.5rem 0 1rem;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#7d8590;">
    🌬️ AQI Forecast · Stasiun Benowo · TFT (pytorch-forecasting) · EXP3 · dropout=0.25 · lr=3e-05 · hidden=64 · lstm_layers=2
  </span>
</div>
""", unsafe_allow_html=True)
