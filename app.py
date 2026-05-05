"""app.py

Production-ready Streamlit interface for the MPV Phenotype Engine.
No scoring or clinical logic lives here — all computation goes through
ClinicalSupportEngine in clinical_support.py.

Run:
    streamlit run app.py
"""
from __future__ import annotations
import base64
import html
import io
import os
import numpy as np
# Last Updated: 2026-04-13 10:28 (Updated to all-HPO April 2026 baseline) # module_all_HPO_background_comparison_20260413_1019
import sys
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import math
from PIL import Image, ImageDraw

from clinical_support import ClinicalSupportEngine
from prediction_engine import ig_qualitative_label

_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">'
    '<rect width="120" height="120" rx="18" fill="#080d14"/>'
    '<circle cx="60" cy="60" r="52" fill="none" stroke="#00c8b4" stroke-width="1"'
    ' stroke-opacity="0.3" stroke-dasharray="4 8"/>'
    '<circle cx="60" cy="60" r="38" fill="none" stroke="#00c8b4" stroke-width="1.4"'
    ' stroke-opacity="0.55" stroke-dasharray="22 6"/>'
    '<circle cx="60" cy="60" r="24" fill="none" stroke="#f59e3a" stroke-width="1.8"'
    ' stroke-opacity="0.7" stroke-dasharray="12 5"/>'
    '<circle cx="60" cy="22" r="3.5" fill="#00c8b4"/>'
    '<circle cx="93" cy="41" r="3" fill="#00c8b4" opacity="0.7"/>'
    '<circle cx="93" cy="79" r="3" fill="#f59e3a"/>'
    '<circle cx="60" cy="98" r="3.5" fill="#f59e3a"/>'
    '<circle cx="27" cy="79" r="3" fill="#f59e3a" opacity="0.7"/>'
    '<circle cx="27" cy="41" r="3" fill="#00c8b4" opacity="0.7"/>'
    '<circle cx="60" cy="60" r="7" fill="#00c8b4"/>'
    '<circle cx="60" cy="60" r="2.5" fill="#fff" opacity="0.95"/>'
    '</svg>'
)
_FAVICON_DATA_URI = (
    "data:image/svg+xml;base64,"
    f"{base64.b64encode(_FAVICON_SVG.encode()).decode()}"
)


def _build_favicon_png() -> io.BytesIO:
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, 128, 128), radius=19, fill="#080d14")

    def _circle(cx, cy, r, outline=None, width=1, fill=None):
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=outline, width=width, fill=fill)

    _circle(64, 64, 55, outline="#285a5a", width=2)
    _circle(64, 64, 41, outline="#168d86", width=2)
    _circle(64, 64, 26, outline="#b56f25", width=3)
    for x, y, r, color in (
        (64, 24, 4, "#00c8b4"),
        (99, 44, 3, "#00c8b4"),
        (99, 84, 3, "#f59e3a"),
        (64, 104, 4, "#f59e3a"),
        (29, 84, 3, "#f59e3a"),
        (29, 44, 3, "#00c8b4"),
    ):
        _circle(x, y, r, fill=color)
    _circle(64, 64, 8, fill="#00c8b4")
    _circle(64, 64, 3, fill="#ffffff")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


_FAVICON_PNG = _build_favicon_png()

# ─────────────────────────────────────────────────────────────────────────────
# Page config — MUST be the first Streamlit call in the file
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MPV Phenotype Engine",
    page_icon=_FAVICON_PNG,
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. Initialize dark mode state in memory
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
st.session_state.dark_mode = False

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Phenotype Query"

if "active_tab_idx" not in st.session_state:
    st.session_state.active_tab_idx = 0

if "_query_observed_pending" not in st.session_state:
    st.session_state._query_observed_pending = None

# ─────────────────────────────────────────────────────────────────────────────
# Favicon — Icon · Compact SVG mark injected as base64 data URI.
# Overrides the emoji page_icon set in set_page_config in all modern browsers.
# ─────────────────────────────────────────────────────────────────────────────
components.html(
    f"""
    <script>
      try {{
        const href = {_FAVICON_DATA_URI!r};
        const doc = window.parent.document;
        doc.querySelectorAll('link[rel~="icon"], link[rel="shortcut icon"]').forEach((el) => el.remove());
        const link = doc.createElement('link');
        link.rel = 'icon';
        link.type = 'image/svg+xml';
        link.href = href;
        doc.head.appendChild(link);
      }} catch (err) {{}}
    </script>
    """,
    height=0,
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — injected once at startup
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* === Sidebar === */
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(175deg, #0a1e33 0%, #1b3d65 100%);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio p,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small { color: #b8d0e8 !important; }

/* ===== NEW: checkbox text always white ===== */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] input:disabled + div + div p {
    color: #ffffff !important;
    opacity: 1 !important;
}
/* ===== END NEW ===== */

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e8f2fb !important; }
[data-testid="stSidebar"] hr { border-color: #2c4f78 !important; }

/* === Main area === */
.block-container { padding-top: 1.2rem !important; max-width: 1200px; }

/* === Typography === */
h1 { color: #0d2137 !important; font-size: 1.85rem !important; margin-bottom: 0.2rem !important; }
h2 { color: #0d2137 !important; font-size: 1.2rem !important;
     border-bottom: 2px solid #1a6b9a; padding-bottom: 5px;
     margin-top: 1.5rem !important; }
h3 { color: #1a6b9a !important; font-size: 1.02rem !important; }

/* === Metric cards === */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #d6e4f0;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(13,33,55,0.07);
    padding: 0.9rem !important;
}
[data-testid="stMetricLabel"] { color: #4a6785 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.04em; }
[data-testid="stMetricValue"] { color: #0d2137 !important; font-size: 1.6rem !important; font-weight: 700 !important; }

/* === Info / intro box === */
.intro-box {
    background: #e8f2fb;
    border-left: 5px solid #1a6b9a;
    border-radius: 0 8px 8px 0;
    padding: 13px 18px;
    margin-bottom: 1.2rem;
    font-size: 0.93rem;
    color: #1d3557;
    line-height: 1.6;
}

/* === Section explanations (italic caption under each subheader) === */
.explain {
    color: #4a6785;
    font-size: 0.84rem;
    margin: -0.35rem 0 0.65rem 0;
    font-style: italic;
}

/* === Next-question / current-question card === */
.q-card {
    background: #ffffff;
    border: 2px solid #1a6b9a;
    border-radius: 10px;
    padding: 16px 22px;
    margin: 10px 0 14px 0;
    box-shadow: 0 2px 10px rgba(26,107,154,0.10);
}
.q-card .q-name { font-size: 1.15rem; font-weight: 700; color: #0d2137; }
.q-card .q-id   { color: #6a8eae; font-size: 0.86rem; margin-left: 8px; }
.q-card .q-ig   { color: #2a9d8f; font-weight: 600; font-size: 0.88rem; margin-top: 6px; }

/* === Session history items === */
.hist-yes  { background:#e8f7f1; border-left:4px solid #2a9d8f;
             border-radius:0 6px 6px 0; padding:6px 12px; margin:3px 0; font-size:0.88rem; }
.hist-no   { background:#fdf2f0; border-left:4px solid #e76f51;
             border-radius:0 6px 6px 0; padding:6px 12px; margin:3px 0; font-size:0.88rem; }
.hist-skip { background:#f5f6fa; border-left:4px solid #adb5bd;
             border-radius:0 6px 6px 0; padding:6px 12px; margin:3px 0; font-size:0.88rem; color:#6c757d; }

/* === Phenotype Chips ===
   .chip.observed uses ✓ prefix; .chip.excluded uses ✕ prefix — color is NOT the only
   differentiator (WCAG 1.4.1 compliant) */
.chip-container { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px; }
/* P3-E: Contrast-corrected chip colors (ratio ≥ 3.0:1 against white text) */
.chip { padding: 4px 10px; border-radius: 16px; font-size: 0.85rem; font-weight: 600; color: #fff;
        min-height: 32px; display: inline-flex; align-items: center; }
.chip.observed { background-color: #207f73; }
.chip.excluded { background-color: #c4553a; }

/* === Workup / Risk styled items (P2-F) === */
.workup-item {
    border-left: 4px solid #1a6b9a;
    background: #e8f2fb;
    border-radius: 0 6px 6px 0;
    padding: 5px 10px;
    margin: 3px 0;
    font-size: 0.9rem;
}
.risk-item {
    border-left: 4px solid #e9922a;
    background: #fff8ee;
    border-radius: 0 6px 6px 0;
    padding: 5px 10px;
    margin: 3px 0;
    font-size: 0.9rem;
}
.next-item {
    border-left: 4px solid #2a9d8f;
    background: #eaf7f5;
    border-radius: 0 6px 6px 0;
    padding: 5px 10px;
    margin: 3px 0;
    font-size: 0.9rem;
}

/* === Touch target sizing (WCAG 2.5.5) === */
/* Streamlit renders chip X-buttons internally; we target the chip container */

/* === Tablet (≤ 900px) === */
@media (max-width: 900px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .chip { font-size: 0.78rem; padding: 5px 10px; min-height: 36px; }
}

/* === Mobile (≤ 480px) === */
@media (max-width: 480px) {
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.0rem !important; }
    .q-card { padding: 10px 14px; }
    .q-card .q-name { font-size: 1.0rem; }
    .chip { font-size: 0.72rem; padding: 6px 10px; min-height: 44px; }
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
}

/* === Print (browser print / Save as PDF) === */
@media print {
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stHeader"] { display: none !important; }
    .block-container { max-width: 100% !important; padding: 0.5rem !important; }
    [data-testid="metric-container"] { box-shadow: none !important; break-inside: avoid; }
    .stApp { background: #ffffff !important; }
}
</style>
""", unsafe_allow_html=True)


# --- 3. Dark Mode Overrides (Only injected if toggle is ON) ---
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    /* ── Backgrounds ─────────────────────────────────────── */
    .stApp, .block-container                     { background-color: #0f172a !important; }
    section[data-testid="stSidebar"] > div       { background: linear-gradient(175deg, #020b18 0%, #0d1f35 100%) !important; }

    /* ── Base text ───────────────────────────────────────── */
    .stApp, .stMarkdown, p, li, label,
    [data-testid="stText"]                       { color: #e2e8f0 !important; }
    h1, h2, h3                                   { color: #f8fafc !important; }
    h2                                           { border-bottom-color: #3b82f6 !important; }

    /* ── Native inputs (multiselect, selectbox, text_input) ─ */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"]                    { background-color: #1e293b !important; border-color: #334155 !important; color: #e2e8f0 !important; }
    [data-baseweb="tag"]                         { background-color: #1a6b9a !important; }
    [data-baseweb="menu"]                        { background-color: #1e293b !important; }
    [data-baseweb="menu"] li:hover               { background-color: #334155 !important; }
    [data-baseweb="menu"] [aria-selected="true"] { background-color: #1a4a6b !important; }

    /* ── Buttons ─────────────────────────────────────────── */
    [data-testid="stBaseButton-secondary"]       { background-color: #1e293b !important; border-color: #334155 !important; color: #e2e8f0 !important; }
    [data-testid="stBaseButton-secondary"]:hover { background-color: #334155 !important; }

    /* ── Tabs ────────────────────────────────────────────── */
    [data-testid="stTabs"] [role="tablist"]      { border-bottom-color: #334155 !important; }
    [data-testid="stTab"]                        { color: #94a3b8 !important; }
    [data-testid="stTab"][aria-selected="true"]  { color: #3b82f6 !important; border-bottom-color: #3b82f6 !important; }

    /* ── Dataframes / tables ─────────────────────────────── */
    [data-testid="stDataFrame"] iframe           { filter: invert(0.9) hue-rotate(180deg); }

    /* ── Expanders ───────────────────────────────────────── */
    [data-testid="stExpander"]                   { border-color: #334155 !important; background: #1e293b !important; }
    [data-testid="stExpander"] summary           { color: #e2e8f0 !important; }

    /* ── Alerts (info / warning / error / success) ───────── */
    [data-testid="stAlert"]                      { background-color: #1e293b !important; color: #e2e8f0 !important; }

    /* ── Metrics ─────────────────────────────────────────── */
    [data-testid="metric-container"]             { background-color: #1e293b !important; border-color: #334155 !important; }
    [data-testid="stMetricValue"]                { color: #f8fafc !important; }
    [data-testid="stMetricLabel"]                { color: #94a3b8 !important; }

    /* ── Custom cards ────────────────────────────────────── */
    .intro-box  { background: #1e293b !important; border-left-color: #3b82f6 !important; color: #e2e8f0 !important; }
    .q-card     { background: #1e293b !important; border-color: #3b82f6 !important; }
    .q-card .q-name { color: #f8fafc !important; }
    .q-card .q-ig   { color: #4ecca3 !important; }
    .explain        { color: #94a3b8 !important; }

    /* ── Workup / Risk / Next items ──────────────────────── */
    .workup-item { background: #0d1f35 !important; border-left-color: #3b82f6 !important; color: #bfdbfe !important; }
    .risk-item   { background: #1c1008 !important; border-left-color: #f59e0b !important; color: #fde68a !important; }
    .next-item   { background: #041f1a !important; border-left-color: #2a9d8f !important; color: #99f6e4 !important; }

    /* ── History items ───────────────────────────────────── */
    .hist-yes  { background: #0d2d1f !important; color: #6ee7b7 !important; }
    .hist-no   { background: #2d0d0d !important; color: #fca5a5 !important; }
    .hist-skip { background: #1e293b !important; color: #94a3b8 !important; }

    /* ── HPO ID code tags ────────────────────────────────── */
    code { background: #1e293b !important; color: #93c5fd !important; border: 1px solid #334155; }

    /* ── Captions & small text ───────────────────────────── */
    [data-testid="stCaptionContainer"] { color: #64748b !important; }

    /* ── Dividers ────────────────────────────────────────── */
    hr { border-color: #334155 !important; }
    </style>
    """, unsafe_allow_html=True)

if st.session_state.get("app_mode") not in {
    "Phenotype Query",
    "Interactive Session",
    "Module Browser",
    "Comparative Analytics",
}:
    st.session_state.app_mode = "Phenotype Query"

st.session_state.dark_mode = False

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
    :root {
        --nav: #071624;
        --nav2: #0d2236;
        --nav-border: #162e48;
        --page: #eef2f8;
        --card: #ffffff;
        --border: #dde4ef;
        --blue: #155fa0;
        --blue-l: #e8f1fc;
        --teal: #0d9b8a;
        --teal-l: #edfaf8;
        --amber: #c97c12;
        --amber-l: #fef4e0;
        --red: #c63030;
        --red-l: #fde8e8;
        --emerald: #0f7b5c;
        --emerald-l: #e6f7f2;
        --ink: #0c1a28;
        --ink2: #3f5672;
        --ink3: #8ca3bc;
        --r: 12px;
        --shadow: 0 1px 3px rgba(7,22,36,.06), 0 4px 16px rgba(7,22,36,.05);
    }
    html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
    }
    .stApp {
        background: var(--page) !important;
        color: var(--ink);
    }
    .block-container {
        max-width: none !important;
        padding-top: 1.25rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 2rem !important;
    }
    h1, h2, h3, h4 {
        font-family: "Outfit", sans-serif !important;
        color: var(--ink) !important;
        border-bottom: none !important;
    }
    code, .mono {
        font-family: "JetBrains Mono", monospace !important;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(170deg, var(--nav) 0%, var(--nav2) 100%) !important;
        border-right: 1px solid var(--nav-border);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    [data-testid="stSidebar"] .sidebar-title {
        font-family: "Outfit", sans-serif;
        font-size: 0.95rem;
        font-weight: 800;
        line-height: 1.15;
        color: #d4eaf8 !important;
    }
    [data-testid="stSidebar"] .sidebar-sub {
        font-size: 0.76rem;
        line-height: 1.5;
        color: #7fa8c4 !important;
    }
    [data-testid="stSidebar"] .sidebar-logo-wrap {
        padding: .3rem 0 .15rem 0;
        margin-bottom: .15rem;
    }
    [data-testid="stSidebar"] .sidebar-logo-wrap svg {
        display: block;
        max-height: 84px;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--nav-border) !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        justify-content: flex-start;
        gap: 0.55rem;
        border-radius: 8px;
        padding: 0.62rem 0.85rem;
        font-size: 0.84rem;
        font-weight: 500;
        box-shadow: none !important;
        border: 1px solid transparent !important;
        transition: background .15s ease, color .15s ease, border-color .15s ease;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: #5e89ab !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: rgba(255,255,255,.06) !important;
        color: #a8c8e2 !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(13,155,138,.14) !important;
        color: #40d9c8 !important;
        border-color: rgba(64,217,200,.18) !important;
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        background: white !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    [data-baseweb="select"] [data-baseweb="tag"] {
        background: #0a4a3a !important;
        color: #5ee8d0 !important;
        border: 1px solid #1a7060 !important;
        border-radius: 20px !important;
        font-weight: 600;
    }
    [data-testid="stTabs"] [role="tablist"] {
        gap: 1rem;
        border-bottom: 1px solid var(--border);
    }
    [data-testid="stTabs"] [role="tab"] {
        height: 42px;
        padding: 0 0.15rem;
        color: var(--ink3);
        background: transparent;
        border-bottom: 2px solid transparent;
        font-size: 0.85rem;
        font-weight: 600;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--blue) !important;
        border-bottom-color: var(--blue) !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--r) !important;
        background: white !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: var(--r);
        overflow: hidden;
        background: white;
        box-shadow: var(--shadow);
    }
    [data-testid="stAlert"] {
        border-radius: var(--r);
        border: 1px solid var(--border);
    }
    .topbar {
        position: sticky;
        top: 0;
        z-index: 25;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.8rem 1rem;
        margin: -0.1rem 0 1.5rem 0;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(238,242,248,.92);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: var(--shadow);
    }
    .topbar .status {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        color: var(--emerald);
        font-size: 0.76rem;
        font-weight: 600;
    }
    .topbar .dot,
    .sidebar-status-dot {
        width: 0.38rem;
        height: 0.38rem;
        border-radius: 999px;
        background: #34d399;
        display: inline-block;
    }
    .page-head {
        margin-bottom: 1.1rem;
    }
    .page-head-inner {
        display: flex;
        align-items: center;
        gap: .75rem;
    }
    .page-head-mark {
        width: 34px;
        height: 34px;
        flex: 0 0 34px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #ffffff;
        border: 1px solid var(--border);
        box-shadow: 0 6px 18px rgba(13,33,55,.08);
    }
    .page-head-mark svg {
        width: 24px;
        height: 24px;
        display: block;
    }
    .page-head h1 {
        font-size: 24px !important;
        font-weight: 800 !important;
        margin: 0 0 0.15rem 0 !important;
    }
    .page-head p {
        margin: 0;
        color: var(--ink2);
        font-size: 14px;
        line-height: 1.6;
    }
    .intro-box,
    .card-shell {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--r);
        box-shadow: var(--shadow);
    }
    .intro-box {
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        color: var(--ink2) !important;
        line-height: 1.65;
        background: linear-gradient(180deg, #f8fbff 0%, #eef5fd 100%) !important;
        border-left: 1px solid var(--border) !important;
    }
    .label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: var(--ink3);
    }
    .section-title {
        margin: .1rem 0 .2rem 0;
        color: var(--ink);
        font-size: 13px;
        font-weight: 800;
        line-height: 1.35;
    }
    .section-subtitle {
        margin: 0 0 .15rem 0;
        color: var(--ink2);
        font-size: 13px;
        font-weight: 650;
        line-height: 1.45;
    }
    .section-copy {
        margin: 0 0 .9rem 0;
        color: var(--ink3);
        font-size: 12px;
        line-height: 1.65;
    }
    .case-load-notice {
        display: flex;
        align-items: center;
        gap: .85rem;
        margin: -.35rem 0 1.05rem 0;
        padding: .75rem .95rem;
        border: 1px solid #bfece4;
        border-left: 4px solid var(--teal);
        border-radius: 10px;
        background: linear-gradient(180deg, #f7fffd 0%, #eefbf8 100%);
        box-shadow: 0 8px 22px rgba(13,33,55,.08);
    }
    .case-load-notice.error {
        border-color: #f3c3bd;
        border-left-color: var(--red);
        background: linear-gradient(180deg, #fffafa 0%, #fff1ef 100%);
    }
    .case-load-mark {
        width: 38px;
        height: 38px;
        flex: 0 0 38px;
        border-radius: 11px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #080d14;
        box-shadow: 0 8px 18px rgba(8,13,20,.18);
    }
    .case-load-mark svg {
        width: 28px;
        height: 28px;
        display: block;
    }
    .case-load-body {
        min-width: 0;
    }
    .case-load-kicker {
        color: var(--teal);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .case-load-title {
        color: var(--ink);
        font-size: 14px;
        font-weight: 800;
        line-height: 1.35;
        margin-top: .08rem;
    }
    .case-load-meta {
        color: var(--ink3);
        font-size: 12px;
        line-height: 1.45;
        margin-top: .12rem;
    }
    .explain-text {
        font-size: 12px;
        line-height: 1.65;
        color: var(--ink3);
        margin: 0.15rem 0 0.9rem 0;
    }
    .soft-note {
        margin: .2rem 0 .1rem 0;
        color: var(--ink3);
        font-size: 12px;
        line-height: 1.55;
    }
    .soft-note strong {
        color: var(--ink);
        font-weight: 700;
    }
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.85rem;
    }
    .chip {
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        padding: 0.28rem 0.7rem;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .chip.observed {
        background: #0a4a3a;
        color: #5ee8d0;
        border: 1px solid #1a7060;
    }
    .chip.excluded {
        background: #4a0e0e;
        color: #f09090;
        border: 1px solid #7a2020;
    }
    .footer-meta {
        margin-top: 1.2rem;
        color: var(--ink3);
        font-size: 12px;
    }
    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    @media (max-width: 640px) {
        .topbar {
            flex-direction: column;
            align-items: flex-start;
        }
        .page-head-inner {
            align-items: flex-start;
        }
        .page-head-mark {
            width: 30px;
            height: 30px;
            flex-basis: 30px;
        }
        .page-head-mark svg {
            width: 21px;
            height: 21px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Engine — loaded once, shared across all Streamlit reruns
# ─────────────────────────────────────────────────────────────────────────────

MODULE_LABELS = {
    0: "Macular Dystrophy / Broad RP",
    1: "Syndromic Optic Atrophy (Mitochondrial)",
    2: "Peroxisomal Syndromic",
    3: "Transcriptional Regulation",
    4: "IFT / Ciliary Dynein (Skeletal Ciliopathy)",
    5: "Melanosome / OCA (Albinism)",
    6: "CSNB / Phototransduction",
    7: "Uncharacterized Syndromic",
    8: "Familial Exudative Vitreoretinopathy (EVR) / Extracellular Matrix",
    9: "Bardet-Biedl Syndrome (BBS) / Ciliopathy",
    10: "Optic Atrophy / Mitochondrial Metabolism",
    11: "Joubert Syndrome / Ciliopathy",
    12: "Leber Congenital Amaurosis (LCA)",
    13: "Syndromic Optic Atrophy (minor)",
    14: "Cone Dystrophy / CRD",
    15: "Color Vision Defects (CVD) / Cone Phototransduction",
    16: "Usher Syndrome",
}

def _module_label(module_id: int) -> str:
    return f"Module {module_id} — {MODULE_LABELS.get(module_id, 'Unknown')}"

_MAX_ENTROPY_NATS = 2.833213344056216    # math.log(17)

def _ig_display(ig: float) -> str:
    """Primary label shown to the clinician — qualitative tier only."""
    if ig <= 0.0:
        return "—  (no data yet)"
    return ig_qualitative_label(ig)


def _ig_tooltip(ig: float) -> str:
    """Raw technical detail for the help= tooltip."""
    if ig <= 0.0:
        return "No information gain calculated (session not yet started)."
    pct = (ig / _MAX_ENTROPY_NATS) * 100
    return (
        f"Raw information gain: {ig:.4f} nats ({pct:.1f}% of maximum possible reduction). "
        "Maximum = ln(17) ≈ 2.833 nats (complete disambiguation across 17 modules). "
        "Thresholds: High ≥ 0.8 nats, Moderate 0.3–0.8 nats, Low < 0.3 nats."
    )

st.markdown(
    f"""
    <style>
    .stSpinner,
    [data-testid="stSpinner"] {{
        position: relative;
        min-height: 76px;
        max-width: 520px;
        margin: .35rem 0 1rem 0;
        padding: 18px 22px 18px 82px;
        border: 1px solid #cfe0ee;
        border-radius: 12px;
        background: linear-gradient(180deg, #f8fbff 0%, #edf5fb 100%);
        box-shadow: 0 12px 30px rgba(13,33,55,.10);
        color: #0d2137;
        overflow: hidden;
    }}
    .stSpinner::before,
    [data-testid="stSpinner"]::before {{
        content: "";
        position: absolute;
        left: 20px;
        top: 50%;
        width: 42px;
        height: 42px;
        transform: translateY(-50%);
        border-radius: 12px;
        background: #080d14 url("{_FAVICON_DATA_URI}") center / 34px 34px no-repeat;
        box-shadow: 0 8px 18px rgba(8,13,20,.20);
    }}
    .stSpinner::after,
    [data-testid="stSpinner"]::after {{
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #009b8c 0%, #d4820a 100%);
    }}
    .stSpinner p,
    [data-testid="stSpinner"] p,
    .stSpinner div,
    [data-testid="stSpinner"] div {{
        color: #0d2137 !important;
        font-family: "Outfit", sans-serif !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        line-height: 1.45 !important;
    }}
    .stSpinner svg:not([aria-hidden="true"]),
    [data-testid="stSpinner"] svg:not([aria-hidden="true"]) {{
        color: #009b8c !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Mapping IRD phenotypes, modules, and gene priors...\nplease wait...")
def _load_engine(gamma: float = 0.3):
    return ClinicalSupportEngine(
        eager=True,
        gamma=gamma,
    )


@st.cache_resource
def _load_discovery_manager(_engine):
    """Build a DiscoveryManager from the engine's EBL matrices (cached by engine identity)."""
    from discovery_manager import EBLSource, DiscoveryManager
    lr = _engine.ebl_lr_matrix
    cnt = _engine.ebl_count_matrix
    if lr is None or cnt is None:
        return None
    return DiscoveryManager([EBLSource(lr, cnt)])



# Sidebar: Engine Parameters (gamma + Ethnicity Bayes Layer)
_ETH_OPTIONS = [
    "",
    "Arab_Muslim",
    "Ashkenazi",
    "Jewish_Other",
    "Middle_Eastern_Jewish",
    "North_African_Jewish",
]

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo-wrap">
          <svg width="100%" viewBox="0 0 220 68" fill="none" xmlns="http://www.w3.org/2000/svg">
            <g transform="translate(5,8) scale(0.42)">
              <circle cx="60" cy="60" r="52" fill="none" stroke="#00c8b4" stroke-width="1.2" stroke-opacity="0.4" stroke-dasharray="4 8"/>
              <circle cx="60" cy="60" r="38" fill="none" stroke="#00c8b4" stroke-width="1.5" stroke-opacity="0.5" stroke-dasharray="22 6"/>
              <circle cx="60" cy="60" r="24" fill="none" stroke="#f59e3a" stroke-width="1.8" stroke-opacity="0.65" stroke-dasharray="12 5"/>
              <circle cx="60" cy="22" r="3.5" fill="#00c8b4"/>
              <circle cx="93" cy="79" r="3" fill="#f59e3a"/>
              <circle cx="27" cy="79" r="3" fill="#f59e3a" opacity="0.7"/>
              <circle cx="60" cy="60" r="6" fill="#00c8b4"/>
              <circle cx="60" cy="60" r="2.5" fill="#fff" opacity="0.9"/>
            </g>
            <text x="72" y="30" font-family="Outfit, sans-serif" font-size="23" font-weight="700" letter-spacing="3.1" fill="#e8f0f8">IRD</text>
            <text x="72" y="50" font-family="DM Mono, monospace" font-size="8.1" font-weight="300" letter-spacing="1.45" fill="#00c8b4">PRIORITIZATION ENGINE</text>
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    nav_options = [
        "Phenotype Query",
        "Interactive Session",
        "Module Browser",
        "Comparative Analytics",
    ]
    if st.session_state.app_mode not in nav_options:
        st.session_state.app_mode = nav_options[0]

    for option in nav_options:
        if st.button(
            option,
            key=f"nav_{option}",
            type="primary" if st.session_state.app_mode == option else "secondary",
            use_container_width=True,
        ):
            st.session_state.app_mode = option
            st.rerun()

    st.divider()

    # Section: Streamlit UI components for Engine Parameters
    st.subheader("Engine Parameters")

    # Displaying markdown with custom HTML formatting
    # Added a subtle horizontal rule (<hr>) after the initial description
    st.markdown(
        """
        <div class="sidebar-sub" style="margin:-.25rem 0 .85rem 0;">
        Controls optional scoring behavior used by the phenotype engine.
        </div>

        <!-- Subtle separator line -->
        <hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.2); margin: 0.5rem 0;">

        <div style="margin:.25rem 0 .12rem 0;font-size:12px;font-weight:700;color:#d4eaf8;">
        Ethnicity Bayes Layer
        </div>
        <div class="sidebar-sub" style="margin-bottom:.45rem;">
        Optional population-aware prior. When enabled, selected ethnicity can adjust gene scores using solved-case likelihood ratios.
        </div>
        """,
        unsafe_allow_html=True,
    )
    eth_group = st.selectbox(
        "Patient ethnicity",
        options=_ETH_OPTIONS,
        format_func=lambda v: "-- Ethnicity Layer OFF --" if v == "" else v.replace("_", " "),
        key="eth_group_sel",
        label_visibility="collapsed",
        help=(
            "Select the patient's ethnic group to enable population-specific "
            "founder-variant enrichment. Applies to both the primary gene ranking "
            "(Track 1) and the Discovery Panel (Track 2)."
        ),
    )
    use_eth_prior = st.checkbox(
        "Enable Ethnicity Prior",
        value=False,
        disabled=(eth_group == ""),
        key="use_eth_prior_cb",
        help=(
            "Multiply each gene's SMA-GS score by its ethnicity likelihood ratio. "
            "Requires an ethnicity selection above. Default off -- must be explicitly enabled."
        ),
    )
    if eth_group == "":
        use_eth_prior = False
    # Keep a canonical key for render paths that still read session ethnicity.
    st.session_state["ethnicity"] = eth_group

    # Render module leakage header with an added separator above it
    st.markdown(
        """
        <!-- Subtle separator line with reduced top margin -->
        <hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.2); margin: 0rem 0 0.5rem 0;">

        <div style="margin:0.25rem 0 .12rem 0;font-size:12px;font-weight:700;color:#d4eaf8;">
        Module leakage (&gamma;)
        </div>
        <div class="sidebar-sub" style="margin-bottom:.45rem;">
        Phenotype evidence sharing. Higher values let disease-module patterns contribute more to genes without direct HPO matches.
        </div>
        """,
        unsafe_allow_html=True,
    )

    gamma_val = st.slider(
        "Module leakage (γ)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        label_visibility="collapsed",
        help=(
            "Controls leakage credit from module symptoms to genes without direct annotations. "
            "γ=0 is strict matching; γ=1 is full module-wide credit for core genes."
        ),
    )

    st.divider()


engine = _load_engine(gamma=gamma_val)


# ─────────────────────────────────────────────────────────────────────────────
# Autocomplete option lists — built once from engine data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def _build_hpo_options(_engine) -> list[str]:
    """Format: 'Term Name  (HP:XXXXXXX)' — name first so it's prominent in search.
    Already sorted by max prevalence descending (P2-A) by get_hpo_options()."""
    return [f"{name}  ({hid})" for hid, name in _engine.get_hpo_options()]

@st.cache_data
def _build_gene_options(_engine) -> list[str]:
    return _engine.get_gene_options()

HPO_OPTIONS  = _build_hpo_options(engine)
GENE_OPTIONS = _build_gene_options(engine)
HPO_OPTION_SET = frozenset(HPO_OPTIONS)

# ─────────────────────────────────────────────────────────────────────────────
# Clinical Case Library (Demo Buttons)
# ─────────────────────────────────────────────────────────────────────────────

# ── Clinical Case Library ────────────────────────────────────────────────────
# Each case is a validated phenotypic profile cross-referenced against
# validation/classic_cases.py. HPO terms are selected for maximum module
# discriminability: prefer terms with high target_module_phenotype_prevalence
# AND high target_module_share_of_phenotype_percent in the xlsx data.
#
# Module coverage: 0 (Macular Dystrophy / Broad RP),
#                  9 (Bardet-Biedl Syndrome (BBS) / Ciliopathy),
#                  12 (Leber Congenital Amaurosis (LCA)),
#                  15 (Color Vision Defects (CVD) / Cone Phototransduction),
#                  16 (Usher Syndrome)
# ─────────────────────────────────────────────────────────────────────────────

CLINICAL_CASES = {
    # Module 9 — BBS / Ciliopathy
    # Discriminating strategy: Hepatic fibrosis (HP:0001395) is the highest-
    # specificity mod-9 marker. Postaxial polydactyly of hand (HP:0001162) is
    # preferred over generic Polydactyly (HP:0010442) as it carries higher
    # module share. Hypertension (HP:0000822) encodes renal involvement.
    # Validated against classic_cases.py → expected_module=9. ✓
    "Bardet-Biedl Syndrome": {
        "hpos": [
            "HP:0000510",  # Rod-cone dystrophy — universal retinal anchor
            "HP:0001513",  # Obesity — shared BBS/Alstrom ciliopathy feature
            "HP:0001162",  # Postaxial polydactyly of hand — BBS-specific (replaces generic HP:0010442)
            "HP:0001395",  # Hepatic fibrosis — highly specific to Module 9
            "HP:0000822",  # Hypertension — renal involvement marker
        ],
        "excluded": [],
        "desc": "Bardet-Biedl Syndrome (BBS) / Ciliopathy (Module 9)",
        "rationale": "High confidence; Hepatic fibrosis and Postaxial polydactyly are the strongest Module 9 discriminators, enabling precise gene-list recovery for this complex pleiotropic ciliopathy."
    },
    # Module 16 — Usher Syndrome Type 1
    # Discriminating strategy: Vestibular hyporeflexia (HP:0001756) is present
    # in 56.25% of mod-16 genes and is nearly exclusive to this module — the
    # single most powerful Usher discriminator. HP:0011389 (Functional
    # abnormality of the inner ear) reinforces the vestibular-cochlear axis.
    # Rod-cone dystrophy (HP:0000510) is the retinal anchor.
    # Validated against classic_cases.py → expected_module=16. ✓
    "Usher Syndrome Type 1": {
        "hpos": [
            "HP:0000510",  # Rod-cone dystrophy — retinal anchor
            "HP:0000407",  # Sensorineural hearing impairment — 87.5% Module 16
            "HP:0001756",  # Vestibular hyporeflexia — 56.25% Module 16, near-exclusive (replaces HP:0001751)
            "HP:0011389",  # Functional abnormality of the inner ear — cochlear/vestibular axis
        ],
        "excluded": [],
        "desc": "Usher Syndrome (Module 16)",
        "rationale": "High confidence; Vestibular hyporeflexia (HP:0001756) is the critical discriminator at 56% Module 16 prevalence, cleanly separating Usher from other hearing-retinal overlaps."
    },
    # Module 12 — Leber Congenital Amaurosis (LCA)
    # Discriminating strategy: Keratoconus (HP:0000563) is a clinically
    # validated LCA marker absent from most other IRD modules. Combined with
    # nystagmus, severe VI, cataract, and photophobia, this produces a
    # high-confidence module-12 profile.
    # Validated against classic_cases.py → expected_module=12. ✓
    "Leber Congenital Amaurosis": {
        "hpos": [
            "HP:0000639",  # Nystagmus — early-onset LCA hallmark
            "HP:0001141",  # Severely reduced visual acuity
            "HP:0000563",  # Keratoconus — LCA-specific structural marker
            "HP:0000518",  # Cataract
            "HP:0000613",  # Photophobia
        ],
        "excluded": [],
        "desc": "Leber Congenital Amaurosis / LCA (Module 12)",
        "rationale": "High-confidence LCA profile using verified markers: Nystagmus, Severe VI, Keratoconus, Cataract, and Photophobia."
    },
    # Module 0 — Isolated RP (Non-Syndromic)
    # Discriminating strategy: Bone spicule pigmentation (HP:0007737) and
    # Attenuation of retinal blood vessels (HP:0007843) are the hallmark RP
    # findings. Constriction of peripheral visual field (HP:0001133) encodes
    # progressive tunnel vision. Explicit exclusions are included to reduce
    # syndromic and cone-only confounders.
    # Validated against classic_cases.py → expected_module=0. ✓
    "Isolated RP (Non-Syndromic)": {
        "hpos": [
            "HP:0000510",  # Rod-cone dystrophy
            "HP:0000662",  # Nyctalopia — early RP symptom
            "HP:0001105",  # Retinal atrophy
            "HP:0000580",  # Pigmentary retinopathy
            "HP:0007737",  # Bone spicule pigmentation — hallmark RP structural sign
            "HP:0007843",  # Attenuation of retinal blood vessels
            "HP:0001133",  # Constriction of peripheral visual field (48% mod 0, 12% mod 15)
            "HP:0007675",  # Progressive night blindness
        ],
        "excluded": [
            "HP:0001513",  # Obesity
            "HP:0001162",  # Polydactyly
            "HP:0000407",  # Hearing loss
            "HP:0001249",  # Intellectual disability
            "HP:0002419",  # Molar tooth sign
            "HP:0007803",  # Monochromacy
            "HP:0008275",  # Abnormal light-adapted ERG
            "HP:0000540",  # Hypermetropia
        ],
        "desc": "Macular Dystrophy / Broad RP (Module 0)",
        "rationale": "High-confidence lead using both hallmark RP findings and targeted exclusions to suppress syndromic/ciliopathy and cone-only alternatives."
    },
    # Module 15 — Color Vision Defects (CVD) / Cone Phototransduction
    # Discriminating strategy: Monochromacy (HP:0007803) is 100% prevalent in
    # Module 15 with 72.7% module share — the single most exclusive IRD term.
    # Undetectable light-adapted ERG (HP:0030465) reinforces cone-specific
    # dysfunction. This case was added to fill a gap in module coverage and
    # replace the prior Ambiguous Case, which lacked a validated HPO profile.
    # Validated against classic_cases.py → expected_module=15. ✓
    "Achromatopsia": {
        "hpos": [
            "HP:0007803",  # Monochromacy — 100% Module 15, 72.7% module share (near-exclusive)
            "HP:0008275",  # Abnormal light-adapted ERG — 87.5% Module 15
            "HP:0030465",  # Undetectable light-adapted ERG — 87.5% Module 15, 100% share
            "HP:0000613",  # Photophobia
            "HP:0000639",  # Nystagmus
        ],
        "excluded": [],
        "desc": "Color Vision Defects (CVD) / Cone Phototransduction (Module 15)",
        "rationale": "High-confidence cone-only disease. Monochromacy and undetectable light-adapted ERG are hallmarks exclusive to Module 15, cleanly distinguishing it from all rod-involving dystrophies."
    },
}

REAL_CASES_CSV = os.path.join("Input", "real_clinical_cases_signal_only_13.04.26.csv")

def _hpo_labels_from_ids(hpo_ids: list[str]) -> list[str]:
    """Resolve HPO IDs into the exact labels used by the phenotype widget."""
    labels = []
    for hid in hpo_ids:
        name = engine.get_term_name(hid)
        label = f"{name}  ({hid})"
        if label in HPO_OPTION_SET:
            labels.append(label)
    return labels

@st.cache_data
def _load_real_clinical_cases() -> list[dict]:
    """Load real clinical cases with at least two valid HPO phenotypes."""
    if not os.path.exists(REAL_CASES_CSV):
        return []

    df = pd.read_csv(REAL_CASES_CSV)
    cases = []

    for row in df.itertuples(index=False):
        raw_hpos = str(getattr(row, "resolved_hpo_ids", "") or "")
        hpo_ids = []
        seen = set()

        for part in raw_hpos.split(";"):
            hid = part.strip()
            if hid.startswith("HP:") and hid not in seen:
                seen.add(hid)
                hpo_ids.append(hid)

        valid_labels = _hpo_labels_from_ids(hpo_ids)
        if len(valid_labels) < 2:
            continue

        case_id = str(getattr(row, "case_id", "") or "").strip()
        gene = str(getattr(row, "causal_gene", "") or "").strip() or "Unknown gene"
        phenotype_raw = str(getattr(row, "phenotype_raw", "") or "").strip()

        cases.append(
            {
                "name": f"Case {case_id} - {gene}",
                "hpos": hpo_ids,
                "valid_labels": valid_labels,
                "desc": phenotype_raw or "Real clinical case",
                "rationale": f"HPO IDs: {', '.join(hpo_ids)}",
            }
        )

    cases.sort(key=lambda case: case["name"])
    return cases

def _populate_query_from_case(hpo_ids: list[str], excluded_ids: list[str] | None = None) -> None:
    """Load HPO IDs into the phenotype query widgets and trigger execution."""
    st.session_state.query_observed = []
    st.session_state.query_excluded = []
    st.session_state.query_gene = ""

    st.session_state.query_observed = _hpo_labels_from_ids(hpo_ids)
    st.session_state.query_excluded = _hpo_labels_from_ids(excluded_ids or [])
    st.session_state.run_demo = True

def _demo_callback(case_name: str):
    """Callback for clinical case buttons: cleanup, load, and trigger run."""
    case_data = CLINICAL_CASES[case_name]

    # 1. State Isolation: clear previous query
    st.session_state.query_observed = []
    st.session_state.query_excluded = []
    st.session_state.query_gene = ""

    # 2. Populate new terms
    new_observed = []
    for hid in case_data["hpos"]:
        name = engine.get_term_name(hid)
        label = f"{name}  ({hid})"
        if label in HPO_OPTION_SET:
            new_observed.append(label)

    new_excluded = []
    for hid in case_data.get("excluded", []):
        name = engine.get_term_name(hid)
        label = f"{name}  ({hid})"
        if label in HPO_OPTION_SET:
            new_excluded.append(label)

    st.session_state.query_observed = new_observed
    st.session_state.query_excluded = new_excluded

    # 3. Trigger immediate execution
    st.session_state.run_demo = True

    # 4. Feedback
    _queue_case_notice(
        "Example Loaded",
        case_name,
        "Phenotype terms inserted. The engine will run this profile automatically.",
    )

def _render_clinical_cases():
    """Render the clinical case demo buttons grid."""
    _render_label("Validated Examples")
    _render_explain("Predefined teaching examples. Choosing one fills the phenotype fields and runs the query.")

    # Use a grid layout for better aesthetics
    cols = st.columns(len(CLINICAL_CASES))
    for i, (name, data) in enumerate(CLINICAL_CASES.items()):
        with cols[i]:
            st.button(
                name,
                key=f"btn_case_{i}",
                help=f"{data['desc']}: {data['rationale']}",
                use_container_width=True,
                on_click=_demo_callback,
                args=(name,)
            )

def _real_case_callback(case_name: str):
    """Callback for real clinical case buttons loaded from CSV."""
    case_data = next((case for case in _load_real_clinical_cases() if case["name"] == case_name), None)
    if case_data is None:
        _queue_case_notice(
            "Case Unavailable",
            "Could not load the selected real clinical case",
            "The source row was not found in the clinical case table.",
            level="error",
        )
        return

    _populate_query_from_case(case_data["hpos"])
    _queue_case_notice(
        "Real Case Loaded",
        case_name,
        f"{len(case_data['valid_labels'])} phenotype terms inserted. The query will run automatically.",
    )

def _render_case_button_grid(cases: list[dict], key_prefix: str, callback, columns_per_row: int = 3):
    """Render a compact multi-row button grid."""
    for start in range(0, len(cases), columns_per_row):
        cols = st.columns(columns_per_row)
        for col, case in zip(cols, cases[start:start + columns_per_row]):
            with col:
                st.button(
                    case["name"],
                    key=f"{key_prefix}_{case['name']}",
                    help=f"{case['desc']}: {case['rationale']}",
                    use_container_width=True,
                    on_click=callback,
                    args=(case["name"],)
                )

def _render_real_clinical_cases():
    """Render real-case buttons sourced from the CSV input file."""
    real_cases = _load_real_clinical_cases()
    if not real_cases:
        return

    _render_section_header(
        "Real Clinical Cases",
        "Patient-derived phenotype profiles",
        "Browse eligible de-identified cases from the local clinical case table. Selecting a case inserts its HPO terms and runs the same query workflow.",
    )

    with st.expander(f"Browse {len(real_cases)} eligible real cases", expanded=False):
        _render_case_button_grid(real_cases, "btn_real_case", _real_case_callback)

def _add_hpo_to_query_observed(hpo_id: str) -> bool:
    """Find the exact label for this ID in HPO_OPTIONS and staging it for addition."""
    # Find exact matching label in the global options list
    target_label = next((opt for opt in HPO_OPTIONS if f"({hpo_id})" in opt), None)

    if not target_label:
        return False

    cur = list(st.session_state.get("query_observed", []))
    if target_label not in cur:
        cur.append(target_label)
        # Use staging key to avoid widget-state clobbering
        st.session_state["_query_observed_pending"] = cur
        st.session_state["auto_run_query"] = True
        st.session_state["query_gene"]     = "" # Clear gene search
        st.session_state["active_tab_idx"] = 2 # Persist to 'Workup and prognosis' tab
    return True

def _hpo_id(formatted: str) -> str:
    """Extracts HPO ID from 'Nystagmus  (HP:0000639)' -> 'HP:0000639'"""
    return formatted.split("(")[-1].rstrip(")")

def _hpo_ids(lst: list[str]) -> list[str]:
    return [_hpo_id(s) for s in lst]

def _get_stability_icon(stability: str) -> str:
    """Returns an emoji indicator with clinician-facing cluster confidence label."""
    stability = stability.lower()
    if stability == "core": return "🟢 High cluster confidence"
    if stability == "peripheral": return "🟡 Moderate cluster confidence"
    if stability == "unstable": return "🔴 Low cluster confidence"
    return stability.capitalize()

def _hpo_search_hint_md() -> str:
    return (
        "Can't find a term? Try searching by HPO ID (e.g. HP:0000639) "
        "or a partial name. Browse the full HPO at [hpo.jax.org](https://hpo.jax.org/)."
    )


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _render_topbar() -> None:
    st.markdown(
        """
        <div class="topbar">
          <div style="display:flex;align-items:center;gap:0.6rem;">
            <svg width="22" height="22" viewBox="0 0 120 120" fill="none"
                 xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;opacity:0.72;">
              <circle cx="60" cy="60" r="52" fill="none" stroke="#009b8c"
                stroke-width="1" stroke-opacity="0.35" stroke-dasharray="4 8"/>
              <circle cx="60" cy="60" r="38" fill="none" stroke="#009b8c"
                stroke-width="1.5" stroke-opacity="0.5" stroke-dasharray="22 6"/>
              <circle cx="60" cy="60" r="24" fill="none" stroke="#d4820a"
                stroke-width="1.8" stroke-opacity="0.65" stroke-dasharray="12 5"/>
              <circle cx="60" cy="22" r="3.5" fill="#009b8c"/>
              <circle cx="93" cy="79" r="3"   fill="#d4820a"/>
              <circle cx="27" cy="79" r="3"   fill="#d4820a" opacity="0.7"/>
              <circle cx="60" cy="60" r="6"   fill="#009b8c"/>
              <circle cx="60" cy="60" r="2.5" fill="#fff" opacity="0.9"/>
            </svg>
            <div class="mono" style="font-size:12px;color:var(--ink3);">
              HPO 2026-04-13 &middot; 442 IRD genes &middot; 17 disease modules &middot; 5 ethnicity layer
            </div>
          </div>
          <div class="status">
            <span class="dot"></span>
            Engine ready
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-head">
          <div class="page-head-inner">
            <div class="page-head-mark" aria-hidden="true">
              <svg viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="60" cy="60" r="52" fill="none" stroke="#009b8c" stroke-width="1.1" stroke-opacity="0.35" stroke-dasharray="4 8"/>
                <circle cx="60" cy="60" r="38" fill="none" stroke="#009b8c" stroke-width="1.5" stroke-opacity="0.55" stroke-dasharray="22 6"/>
                <circle cx="60" cy="60" r="24" fill="none" stroke="#d4820a" stroke-width="1.8" stroke-opacity="0.7" stroke-dasharray="12 5"/>
                <circle cx="60" cy="22" r="3.5" fill="#009b8c"/>
                <circle cx="93" cy="79" r="3" fill="#d4820a"/>
                <circle cx="27" cy="79" r="3" fill="#d4820a" opacity="0.7"/>
                <circle cx="60" cy="60" r="6" fill="#009b8c"/>
                <circle cx="60" cy="60" r="2.5" fill="#fff" opacity="0.9"/>
              </svg>
            </div>
            <div>
              <h1>{_esc(title)}</h1>
              <p>{_esc(subtitle)}</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _queue_case_notice(kicker: str, title: str, meta: str, level: str = "success") -> None:
    st.session_state["case_load_notice"] = {
        "kicker": kicker,
        "title": title,
        "meta": meta,
        "level": level,
    }


def _render_case_load_notice() -> None:
    notice = st.session_state.pop("case_load_notice", None)
    if not notice:
        return
    level_class = " error" if notice.get("level") == "error" else ""
    st.markdown(
        f"""
        <div class="case-load-notice{level_class}">
          <div class="case-load-mark" aria-hidden="true">
            <svg viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="60" cy="60" r="52" fill="none" stroke="#00c8b4" stroke-width="1.2" stroke-opacity="0.35" stroke-dasharray="4 8"/>
              <circle cx="60" cy="60" r="38" fill="none" stroke="#00c8b4" stroke-width="1.5" stroke-opacity="0.55" stroke-dasharray="22 6"/>
              <circle cx="60" cy="60" r="24" fill="none" stroke="#f59e3a" stroke-width="1.8" stroke-opacity="0.7" stroke-dasharray="12 5"/>
              <circle cx="60" cy="22" r="3.5" fill="#00c8b4"/>
              <circle cx="93" cy="79" r="3" fill="#f59e3a"/>
              <circle cx="27" cy="79" r="3" fill="#f59e3a" opacity="0.7"/>
              <circle cx="60" cy="60" r="6" fill="#00c8b4"/>
              <circle cx="60" cy="60" r="2.5" fill="#fff" opacity="0.95"/>
            </svg>
          </div>
          <div class="case-load-body">
            <div class="case-load-kicker">{_esc(notice.get("kicker", ""))}</div>
            <div class="case-load-title">{_esc(notice.get("title", ""))}</div>
            <div class="case-load-meta">{_esc(notice.get("meta", ""))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_label(text: str) -> None:
    st.markdown(f'<div class="label">{_esc(text)}</div>', unsafe_allow_html=True)


def _render_section_header(title: str, subtitle: str, body: str = "") -> None:
    body_html = f'<p class="section-copy">{_esc(body)}</p>' if body else ""
    st.markdown(
        f"""
        <div class="section-title">{_esc(title)}</div>
        <div class="section-subtitle">{_esc(subtitle)}</div>
        {body_html}
        """,
        unsafe_allow_html=True,
    )


def _render_explain(text: str) -> None:
    st.markdown(f'<p class="explain-text">{_esc(text)}</p>', unsafe_allow_html=True)


def _confidence_color(value: float) -> str:
    if value > 0.85:
        return "var(--teal)"
    if value > 0.60:
        return "var(--amber)"
    return "var(--red)"


def _ig_style(ig: float) -> tuple[str, str, str]:
    if ig >= 0.8:
        return "High", "#d1fae5", "#065f46"
    if ig >= 0.3:
        return "Moderate", "#fef3c7", "#854d0e"
    return "Low", "#f0f4f8", "#4a6785"


def _stability_style(stability: str) -> dict[str, str]:
    key = stability.lower()
    mapping = {
        "core": {"label": "High confidence", "bg": "#e0f7f0", "color": "#0a6647", "dot": "High"},
        "peripheral": {"label": "Moderate confidence", "bg": "#fef9c3", "color": "#854d0e", "dot": "Moderate"},
        "unstable": {"label": "Low confidence", "bg": "#ffe4e4", "color": "#9b2c2c", "dot": "Low"},
    }
    return mapping.get(key, mapping["peripheral"])


def _render_conf_gauge(confidence: float, caption: str = "Confidence") -> None:
    radius = 38
    circumference = 2 * math.pi * radius
    stroke = _confidence_color(confidence)
    st.markdown(
        f"""
        <div class="card-shell" style="padding:1rem;display:flex;align-items:center;justify-content:center;height:100%;">
          <svg width="96" height="96" viewBox="0 0 96 96" role="img" aria-label="{_esc(caption)} {confidence * 100:.1f}%">
            <circle cx="48" cy="48" r="{radius}" fill="none" stroke="#e8eef6" stroke-width="7"></circle>
            <circle cx="48" cy="48" r="{radius}" fill="none" stroke="{stroke}" stroke-width="7"
              stroke-linecap="round" stroke-dasharray="{circumference:.3f}"
              stroke-dashoffset="{circumference * (1 - confidence):.3f}"
              transform="rotate(-90 48 48)"
              style="transition:stroke-dashoffset .9s cubic-bezier(.16,1,.3,1);"></circle>
            <text x="48" y="44" text-anchor="middle" dominant-baseline="central"
              style="font-size:19px;font-weight:800;fill:var(--ink);font-family:Outfit,sans-serif;">{round(confidence * 100)}%</text>
            <text x="48" y="63" text-anchor="middle"
              style="font-size:8px;fill:var(--ink3);font-family:DM Sans,sans-serif;letter-spacing:.08em;text-transform:uppercase;">{_esc(caption)}</text>
          </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_top_module_hero(result) -> None:
    tm = result.top_module
    supporting = tm.supporting_phenotypes[:4]
    support_text = ", ".join(supporting) if supporting else "Posterior peak after phenotype matching."
    st.markdown(
        f"""
        <div style="position:relative;overflow:hidden;border-radius:14px;padding:1.25rem;color:white;background:linear-gradient(135deg,#0a3258 0%,#0c6b5d 100%);box-shadow:0 6px 24px rgba(13,155,138,.22);">
          <div style="position:absolute;right:-2rem;top:-2rem;width:8rem;height:8rem;border-radius:999px;background:rgba(255,255,255,.10);"></div>
          <div style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:rgba(255,255,255,.6);margin-bottom:.35rem;">Top module</div>
          <div style="font-family:Outfit,sans-serif;font-size:1.6rem;font-weight:800;line-height:1.1;">Module {tm.module_id}</div>
          <div style="font-size:0.92rem;color:rgba(255,255,255,.82);font-weight:500;margin:.35rem 0 1rem 0;">{_esc(MODULE_LABELS.get(tm.module_id, "Unknown"))}</div>
          <div style="display:flex;gap:1.4rem;">
            <div>
              <div style="font-family:Outfit,sans-serif;font-size:1.6rem;font-weight:800;">{tm.probability * 100:.1f}%</div>
              <div style="font-size:11px;color:rgba(255,255,255,.55);">Probability</div>
            </div>
            <div style="padding-left:1.4rem;border-left:1px solid rgba(255,255,255,.2);">
              <div style="font-family:Outfit,sans-serif;font-size:1.6rem;font-weight:800;">{len(result.candidate_genes)}</div>
              <div style="font-size:11px;color:rgba(255,255,255,.55);">Candidate genes</div>
            </div>
          </div>
          <div style="margin-top:.95rem;font-size:12px;color:rgba(255,255,255,.72);line-height:1.55;">
            { _esc(support_text) }
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_module_chart(result, top_n: int | None = None, title: str | None = None) -> None:
    modules = sorted(result.all_modules, key=lambda mm: mm.probability, reverse=True)
    if top_n is not None:
        modules = modules[:top_n]
    max_prob = max((mm.probability for mm in modules), default=1.0) or 1.0
    prior_left = (1 / 17) / max_prob * 100
    rows: list[str] = []
    for mm in modules:
        is_top = mm.module_id == result.top_module.module_id
        width = max(mm.probability / max_prob * 100, 0.5)
        background = "linear-gradient(90deg,#0d9b8a,#1275c0)" if is_top else "#dde4ef"
        prob_text = f"{mm.probability * 100:.1f}%" if mm.probability >= 0.01 else f"{mm.probability * 100:.2f}%"
        prior_line = (
            f'<div style="position:absolute;left:{prior_left:.2f}%;top:0;bottom:0;width:1px;background:#f87171;opacity:.7;"></div>'
            if top_n is None
            else ""
        )
        rows.append(
            f"""
            <div title="{_esc(MODULE_LABELS.get(mm.module_id, 'Unknown'))}" style="display:flex;align-items:center;gap:.7rem;padding:.2rem .2rem;">
              <span class="mono" style="width:1.5rem;font-size:12px;color:var(--ink3);text-align:right;">{mm.module_id}</span>
              <div style="flex:1;position:relative;height:{14 if top_n else 18}px;background:{'#eef2f8' if top_n else 'transparent'};border-radius:4px;overflow:hidden;">
                <div style="height:100%;width:{width:.2f}%;background:{background};border-radius:4px;opacity:{1 if is_top else .9};"></div>
                {prior_line}
              </div>
              <span class="mono" style="width:{46 if top_n else 60}px;font-size:12px;text-align:right;color:{'var(--teal)' if is_top else 'var(--ink3)'};font-weight:{700 if is_top else 500};">{prob_text}</span>
            </div>
            """
        )
    title_html = f'<div class="label" style="margin-bottom:.45rem;">{_esc(title)}</div>' if title else ""
    legend_html = ""
    if top_n is None:
        legend_html = """
        <div style="display:flex;gap:1rem;align-items:center;margin-top:.6rem;">
          <div style="display:flex;gap:.4rem;align-items:center;font-size:12px;color:var(--ink3);">
            <span style="width:1.6rem;height:.45rem;border-radius:4px;background:linear-gradient(90deg,#0d9b8a,#1275c0);display:inline-block;"></span>
            Top module
          </div>
          <div style="display:flex;gap:.4rem;align-items:center;font-size:12px;color:var(--ink3);">
            <span style="width:1px;height:.75rem;background:#f87171;display:inline-block;"></span>
            Prior (1/17 = 5.9%)
          </div>
        </div>
        """
    st.html(
        f"""
        <div class="card-shell" style="padding:1rem 1rem .9rem 1rem;">
          {title_html}
          {''.join(rows)}
          {legend_html}
        </div>
        """
    )


def _render_next_questions(questions: list) -> None:
    cards: list[str] = []
    for idx, question in enumerate(questions, start=1):
        tier, bg, color = _ig_style(question.information_gain)
        bar_width = min(question.information_gain / _MAX_ENTROPY_NATS * 100, 100)
        cards.append(
            f"""
            <div style="padding:1rem;border:1px solid var(--border);border-radius:12px;background:white;margin-bottom:.7rem;">
              <div style="display:flex;align-items:flex-start;gap:.75rem;margin-bottom:.75rem;">
                <div style="width:1.5rem;height:1.5rem;border-radius:999px;background:#e0f0fa;color:var(--blue);display:flex;align-items:center;justify-content:center;font-family:Outfit,sans-serif;font-size:12px;font-weight:700;flex-shrink:0;">{idx}</div>
                <div style="flex:1;min-width:0;">
                  <div style="font-size:14px;font-weight:600;color:var(--ink);line-height:1.45;">{_esc(question.term_name)}</div>
                  <div class="mono" style="font-size:12px;color:var(--ink3);margin-top:.2rem;">{_esc(question.hpo_id)}</div>
                </div>
                <span style="font-size:12px;font-weight:700;padding:.25rem .55rem;border-radius:999px;background:{bg};color:{color};white-space:nowrap;">{tier} IG</span>
              </div>
              <div style="padding-left:2.25rem;">
                <div style="display:flex;gap:.65rem;align-items:center;">
                  <div style="flex:1;height:5px;border-radius:999px;background:#eef2f8;overflow:hidden;">
                    <div style="width:{bar_width:.2f}%;height:100%;border-radius:999px;background:{color};"></div>
                  </div>
                  <span class="mono" style="width:68px;text-align:right;font-size:12px;font-weight:700;color:{color};">{question.information_gain:.3f} nats</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:.3rem;font-size:11px;color:var(--ink3);">
                  <span>0</span>
                  <span>max = ln(17) = 2.833 nats</span>
                </div>
              </div>
            </div>
            """
        )
    st.html("".join(cards))


def _render_gene_breakdown(gene) -> None:
    breakdown = [(name, contrib) for name, contrib in gene.score_breakdown if contrib > 0]
    max_contrib = max((value for _, value in breakdown), default=0.001)
    stability = _stability_style(gene.stability)
    left_items = []
    for name, contrib in breakdown:
        left_items.append(
            f"""
            <div style="display:flex;align-items:center;gap:.55rem;margin-top:.55rem;">
              <div style="flex:1;font-size:12px;color:var(--ink2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{_esc(name)}</div>
              <div style="width:80px;height:5px;border-radius:999px;background:#dde4ef;overflow:hidden;">
                <div style="width:{contrib / max_contrib * 100:.1f}%;height:100%;background:var(--blue);"></div>
              </div>
              <div class="mono" style="width:56px;text-align:right;font-size:12px;font-weight:600;color:var(--blue);">+{contrib:.4f}</div>
            </div>
            """
        )

    stab_html = ""
    if gene.stability_breakdown:
        _, psi, mod = gene.stability_breakdown
        stab_html = f"""
        <div style="margin-top:.65rem;padding:.75rem;border-radius:12px;border:1px solid var(--border);background:{stability['bg']};">
          <div style="font-size:14px;font-weight:700;color:{stability['color']};">{_esc(stability['label'])} (ψ={psi:.2f})</div>
          <div style="display:flex;justify-content:space-between;margin-top:.45rem;font-size:12px;color:var(--ink3);">
            <span>Module-aware credit (imputed)</span>
            <span class="mono" style="font-weight:700;color:{stability['color']};">+{mod:.4f}</span>
          </div>
        </div>
        """

    leak_html = ""
    if gene.leak_breakdown:
        leak_rows = []
        for name, contrib in gene.leak_breakdown:
            leak_rows.append(
                f"""
                <div style="display:flex;align-items:center;gap:.55rem;margin-top:.45rem;">
                  <div style="flex:1;font-size:11px;color:var(--ink3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{_esc(name)}</div>
                  <div class="mono" style="width:56px;text-align:right;font-size:11px;font-weight:600;color:var(--ink3);">+{contrib:.4f}</div>
                </div>
                """
            )
        leak_html = f"""
        <div style="margin-top:1rem;">
          <div class="label" style="font-size:10px;">Module-extrapolated (leak)</div>
          {''.join(leak_rows)}
        </div>
        """

    sig_html = ""
    if gene.supporting_phenotypes:
        best_match = gene.supporting_phenotypes[0]
        if engine.is_module_signature(_find_hpo_id_by_term(best_match), st.session_state.get("current_top_module_id", -1)):
            sig_html = '<div style="margin-top:.55rem;display:inline-flex;gap:.35rem;align-items:center;font-size:12px;font-weight:600;padding:.2rem .55rem;border-radius:999px;background:#fff7e0;color:#a16207;">Signature-linked best match</div>'

    # --- Ethnicity Bayes Layer card ---
    ebl_html = ""
    eth_lr_eff = getattr(gene, "ethnicity_lr_effective", getattr(gene, "ethnicity_lr", None))
    if eth_lr_eff is not None:
        # Determine styling based on the *effective* LR
        lr_color = "var(--teal)" if eth_lr_eff > 1.2 else "var(--red)" if eth_lr_eff < 0.85 else "var(--ink3)"

        rp1l1_warning = ""
        # Prefer the active sidebar selection key, with legacy fallback.
        active_ethnicity = st.session_state.get("eth_group_sel", "") or st.session_state.get("ethnicity", "")
        if gene.gene == "RP1L1" and active_ethnicity == "Ashkenazi":
            rp1l1_warning = (
                '<div style="margin-top:.45rem;padding:.3rem .6rem;border-radius:6px;'
                'background:#fef3c7;color:#b45309;font-size:11px;line-height:1.5;">'
                '<b>Note:</b> LR may be underestimated. Compound-het founder variants '
                '(c.6041A&gt;G + c.6512A&gt;G) are currently stored below the LP threshold '
                'in the source DB and are absent from the training set.'
                '</div>'
            )
        eth_display = active_ethnicity.replace("_", " ") if active_ethnicity else "Unknown"
        base_score = gene.score / eth_lr_eff if eth_lr_eff != 1.0 else gene.score

        x_val = getattr(gene, "ethnicity_count_n", None) or 0
        y_val = engine.ebl_ethnicity_totals.get(active_ethnicity, 0)

        count_html = ""
        enrichment_html = ""

        if y_val > 0 and active_ethnicity:
            count_html = f'<div style="margin-top:.4rem;font-size:11px;color:var(--ink3);">Appears in <b>{int(x_val)}</b> out of <b>{y_val}</b> cases for this ethnicity.</div>'

            # Calculate enrichment vs. other ethnic groups
            cnt_matrix = engine.ebl_count_matrix
            if cnt_matrix is not None and gene.gene in cnt_matrix.index:
                other_eths = [e for e in cnt_matrix.columns if e != active_ethnicity]
                x_other = sum(float(cnt_matrix.at[gene.gene, e]) for e in other_eths if e in cnt_matrix.columns)
                y_other = sum(engine.ebl_ethnicity_totals.get(e, 0) for e in other_eths)

                if y_other > 0:
                    p_target = (x_val / y_val) * 100
                    p_other = (x_other / y_other) * 100
                    rr = (x_val / y_val) / (x_other / y_other) if p_other > 0 else float('inf')

                    if rr == float('inf'):
                        enrichment_text = f"Appears in <b>{p_target:.1f}%</b> of {active_ethnicity.replace('_', ' ')} patients, but <b>not found</b> in other ethnic groups studied."
                    elif rr > 1.2:
                        enrichment_text = f"<b>{p_target:.1f}%</b> of {active_ethnicity.replace('_', ' ')} cases vs. <b>{p_other:.2f}%</b> in other groups. <b>Enrichment: {rr:.1f}× higher</b>."
                    else:
                        enrichment_text = f"<b>{p_target:.1f}%</b> of {active_ethnicity.replace('_', ' ')} cases vs. <b>{p_other:.2f}%</b> in other groups.<br>Similar frequency across ethnicities."

                    enrichment_html = f'<div style="margin-top:.3rem;padding:.35rem .5rem;border-radius:6px;background:#f0fdf4;border-left:3px solid #22c55e;font-size:11px;color:var(--ink2);line-height:1.4;">{enrichment_text}</div>'

        reason_code = getattr(gene, "ethnicity_rule_reason", None) or "unknown"
        raw_lr = getattr(gene, "ethnicity_lr_raw", None) or eth_lr_eff or 1.0

        if reason_code == "downweight_prevented_by_policy":
            policy_explainer = "Downweighting is disabled by policy, so this did not reduce the score."
        elif reason_code == "lr_below_boost_threshold":
            policy_explainer = "Evidence signal is below the partial/boost threshold, so this was not applied."
        elif reason_code.startswith("insufficient_evidence_n_lt_"):
            min_n = reason_code.rsplit("_", 1)[-1]
            policy_explainer = f"Too few training cases for a reliable boost (requires at least {min_n} cases)."
        elif reason_code.startswith("insufficient_evidence_partial_n_lt_"):
            min_n = reason_code.rsplit("_", 1)[-1]
            policy_explainer = f"Signal is moderate, but there are too few training cases for a partial boost (requires at least {min_n} cases)."
        elif reason_code == "partial_boost_applied":
            policy_explainer = "Moderate signal: a partial boost was applied to the final score."
        elif reason_code == "boost_applied":
            policy_explainer = "Boost applied to the final score."
        else:
            policy_explainer = "Policy decision applied."

        reason_html = ""
        if eth_lr_eff != raw_lr:
            reason_html = (
                f'<div style="margin-top:.2rem;font-size:10px;color:var(--ink3);">'
                f'Raw ethnicity evidence: <b>LR {raw_lr:.2f}</b>.<br>{policy_explainer}'
                f'</div>'
            )

        ebl_html = f"""
        <div style="margin-top:.65rem;padding:.75rem;border-radius:12px;border:1px solid var(--border);background:white;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:.5rem;">
            <div>
              <div class="label">Ethnicity Bayes Layer</div>
              <div style="font-size:11px;color:var(--ink3);margin-top:.15rem;">{_esc(eth_display)}</div>
            </div>
            <div class="mono" style="font-size:1.1rem;font-weight:800;color:{lr_color};">×{eth_lr_eff:.3f}</div>
          </div>
          {count_html}
          {enrichment_html}
          {reason_html}
          {f'<div style="display:flex;justify-content:space-between;margin-top:.4rem;font-size:11px;color:var(--ink3);"><span>SMA-GS base</span><span class="mono">{base_score:.4f}</span></div>' if eth_lr_eff != 1.0 else ""}
          {rp1l1_warning}
        </div>
        """

    # Score label reflects whether EBL was applied
    score_label = "SMA-GS × ethnicity LR" if eth_lr_eff is not None and eth_lr_eff != 1.0 else "phenotype overlap + module credit"

    st.html(
        f"""
        <div class="card-shell" style="padding:1rem;background:#f4f8fc;">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;">
            <div>
              <div class="label">Phenotype Contributions To Score</div>
              {''.join(left_items) if left_items else '<div style="margin-top:.65rem;font-size:12px;color:var(--ink3);">No direct phenotype overlap.</div>'}
              {leak_html}
            </div>
            <div>
              <div class="label">Cluster Stability Modifier</div>
              {stab_html}
              {ebl_html}
              <div style="margin-top:.7rem;padding:.75rem;border-radius:12px;border:1px solid var(--border);background:white;">
                <div style="font-size:12px;color:var(--ink3);">{score_label}</div>
                <div style="display:flex;align-items:baseline;gap:.45rem;margin-top:.25rem;">
                  <span style="font-family:Outfit,sans-serif;font-size:1.35rem;font-weight:800;color:var(--ink);">{gene.score:.4f}</span>
                  <span style="font-size:12px;color:var(--ink3);">final score</span>
                </div>
                {sig_html}
              </div>
            </div>
          </div>
        </div>
        """
    )


def _find_hpo_id_by_term(term_name: str) -> str:
    for option in HPO_OPTIONS:
        if option.startswith(f"{term_name}  ("):
            return _hpo_id(option)
    return term_name


# ─────────────────────────────────────────────────────────────────────────────
# Track 2 — Discovery Panel
# ─────────────────────────────────────────────────────────────────────────────

@st.dialog("Discovery Panel — Exploratory Candidates", width="large")
def _show_discovery_dialog(suggestions: list, ethnicity_group: str) -> None:
    """Modal dialog for Track 2 discovery candidates."""
    from discovery_manager import PLANNED_SOURCES

    eth_display = ethnicity_group.replace("_", " ")
    st.markdown(
        f"""
        <div style="padding:.75rem 1rem;border-radius:10px;background:#fef3c7;border:1px solid #fcd34d;
             color:#92400e;font-size:13px;line-height:1.6;margin-bottom:1rem;">
          <b>Exploratory / Advisory only.</b><br>
          These genes are <em>not</em> part of the primary
          IRD module set. They are surfaced because they show statistically significant ethnic
          enrichment in the <b>{_esc(eth_display)}</b> clinical cohort.<br>
          Results should be interpreted alongside clinical phenotype and variant evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Live source: EBL candidates ──────────────────────────────────────────
    ebl_suggestions = [s for s in suggestions if "EBL" in s.sources]
    if ebl_suggestions:
        st.markdown(
            '<div class="label" style="font-size:12px;margin-bottom:.5rem;">Ethnicity Bayes Layer</div>',
            unsafe_allow_html=True,
        )
        for sug in ebl_suggestions:
            meta = sug.source_metadata.get("EBL", {})
            lr_val = meta.get("lr", 0.0)
            n_val = meta.get("n", 0)
            lr_color = "var(--teal)" if lr_val > 2.5 else "var(--blue)"
            master_badge = (
                '<span style="margin-left:.5rem;font-size:10px;font-weight:700;padding:.15rem .45rem;'
                'border-radius:999px;background:#fef9c3;color:#854d0e;">Master Candidate</span>'
                if sug.is_master_candidate else ""
            )
            st.markdown(
                f"""
                <div class="card-shell" style="padding:.85rem 1rem;margin-bottom:.5rem;display:flex;
                     justify-content:space-between;align-items:center;gap:1rem;">
                  <div>
                    <div class="mono" style="font-size:15px;font-weight:800;color:var(--ink);">
                      {_esc(sug.gene)}{master_badge}
                    </div>
                    <div style="font-size:12px;color:var(--ink3);margin-top:.2rem;">
                      Source: Ethnicity Bayes Layer &nbsp;·&nbsp;
                      Training cases: <b>{n_val}</b> &nbsp;·&nbsp;
                      Population: {_esc(eth_display)}
                    </div>
                  </div>
                  <div style="text-align:right;">
                    <div class="label" style="font-size:10px;">Ethnicity LR</div>
                    <div class="mono" style="font-size:1.25rem;font-weight:800;color:{lr_color};">
                      ×{lr_val:.2f}
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info(f"No EBL candidates meeting the Expert Gate (LR ≥ 2.0, n ≥ 5) for {eth_display}.")

    # ── Planned sources (roadmap stubs) ──────────────────────────────────────
    st.markdown(
        '<div style="height:.75rem;"></div>'
        '<div class="label" style="font-size:12px;margin-bottom:.5rem;">Planned Sources (Future Integration)</div>',
        unsafe_allow_html=True,
    )
    for src in PLANNED_SOURCES:
        st.markdown(
            f"""
            <div class="card-shell" style="padding:.85rem 1rem;margin-bottom:.5rem;
                 opacity:.55;border-style:dashed;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <div class="mono" style="font-size:13px;font-weight:700;color:var(--ink2);">
                    {_esc(src['name'])} — {_esc(src['label'])}
                  </div>
                  <div style="font-size:12px;color:var(--ink3);margin-top:.25rem;line-height:1.5;">
                    {_esc(src['description'])}
                  </div>
                </div>
                <span style="font-size:10px;font-weight:700;padding:.2rem .55rem;border-radius:999px;
                      background:#f1f5f9;color:#64748b;white-space:nowrap;">Planned</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_discovery_badge(suggestions: list, ethnicity_group: str) -> None:
    """Render a compact badge that opens the discovery dialog when clicked."""
    n = len(suggestions)
    eth_display = ethnicity_group.replace("_", " ")
    badge_cols = st.columns([3, 1])
    with badge_cols[0]:
        st.markdown(
            f"""
            <div style="padding:.65rem 1rem;border-radius:10px;border:1px solid #fcd34d;
                 background:#fffbeb;color:#92400e;font-size:13px;font-weight:600;
                 display:flex;align-items:center;gap:.55rem;">
              <span style="font-size:16px;">&#128300;</span>
              <span>{n} discovery candidate{'s' if n != 1 else ''} found
              for <b>{_esc(eth_display)}</b> — not in primary gene set</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with badge_cols[1]:
        if st.button("View Discovery Panel", use_container_width=True, key="open_discovery_dialog"):
            _show_discovery_dialog(suggestions, ethnicity_group)


def _replay_session_confidences(history: list[tuple[str, str, str]]) -> list[float]:
    replay = engine.new_session(
        ethnicity_group=eth_group,
        use_ethnicity_prior=use_eth_prior,
    )
    confidences: list[float] = []
    for hpo_id, _, answer in history:
        if answer == "yes":
            replay.answer_yes(hpo_id)
        elif answer == "no":
            replay.answer_no(hpo_id)
        else:
            if hpo_id not in replay.excluded:
                replay.excluded.append(hpo_id)
        if replay.observed or replay.excluded:
            confidences.append(replay.get_current_result().confidence)
    return confidences


def _render_confidence_tracker(confidences: list[float]) -> None:
    if not confidences:
        return
    steps: list[str] = []
    for idx, confidence in enumerate(confidences):
        color = "var(--teal)" if confidence > 0.85 else "var(--amber)" if confidence > 0.60 else "var(--blue)"
        if idx > 0:
            prev = confidences[idx - 1]
            connector = color if confidence > prev else "#dde4ef"
            steps.append(f'<div style="flex:1;height:2px;background:{connector};"></div>')
        delta_html = '<span style="font-size:11px;color:var(--ink3);">start</span>'
        if idx > 0:
            delta = confidence - confidences[idx - 1]
            delta_html = f'<span style="font-size:11px;font-weight:700;color:{"var(--teal)" if delta >= 0 else "var(--red)"};">{delta:+.1%}</span>'
        steps.append(
            f"""
            <div style="display:flex;flex-direction:column;align-items:center;gap:.25rem;">
              <div style="width:28px;height:28px;border-radius:999px;border:2px solid {color};display:flex;align-items:center;justify-content:center;background:white;color:{color};font-size:10px;font-weight:700;font-family:Outfit,sans-serif;">{round(confidence * 100)}</div>
              {delta_html}
            </div>
            """
        )
    st.html(
        f"""
        <div class="card-shell" style="padding:1rem;">
          <div class="label">Diagnostic Confidence Progression</div>
          <div style="display:flex;align-items:center;gap:0;margin-top:.8rem;">{''.join(steps)}</div>
        </div>
        """
    )


def _render_workup_column(
    title: str,
    subtitle: str,
    items: list,
    accent_color: str,
    accent_bg: str,
    accent_border: str,
    icon: str = "",
    add_prefix: str | None = None,
) -> None:
    st.markdown(
        f"""
        <div class="card-shell" style="padding:1rem;background:{accent_bg};border-color:{accent_border};">
          <div style="font-family:Outfit,sans-serif;font-size:14px;font-weight:700;color:{accent_color};">{_esc((icon + ' ' if icon else '') + title)}</div>
          <div style="font-size:12px;line-height:1.6;color:var(--ink2);margin-top:.35rem;">{_esc(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not items:
        st.markdown('<div style="font-size:12px;color:var(--ink3);padding:.75rem 0;">None identified.</div>', unsafe_allow_html=True)
        return
    for index, term in enumerate(items):
        pct = f"{term.prevalence * 100:.0f}%" if term.prevalence else ""
        item_html = (
            f"""
            <div style="padding:.8rem 0;border-bottom:{'1px solid #eef2f8' if index < len(items) - 1 else 'none'};">
              <div style="font-size:14px;font-weight:600;color:var(--ink);line-height:1.45;">{_esc(term.term_name)}</div>
              <div style="display:flex;gap:.55rem;align-items:center;margin-top:.25rem;">
                <span class="mono" style="font-size:12px;color:var(--ink3);">{_esc(term.hpo_id)}</span>
                <span style="font-size:12px;font-weight:700;color:{accent_color};">{pct}</span>
              </div>
            </div>
            """
        )
        if add_prefix:
            left_col, right_col = st.columns([4, 1.25])
            with left_col:
                st.markdown(item_html, unsafe_allow_html=True)
            with right_col:
                st.button(
                    "+ Add",
                    key=f"{add_prefix}_{term.hpo_id}_{index}",
                    use_container_width=True,
                    on_click=_add_hpo_to_query_observed,
                    args=(term.hpo_id,),
                )
        else:
            st.markdown(item_html, unsafe_allow_html=True)


def _render_gene_table(genes: list, observed_hpo_ids: list[str], table_key: str = "genes") -> None:
    state_key = f"expanded_gene_{table_key}"
    filter_key = f"filter_core_{table_key}"
    sort_key = f"gene_sort_{table_key}"
    core_only = st.checkbox("High confidence only", value=False, key=filter_key)
    sort_by = st.selectbox(
        "Sort by",
        options=["Score", "% Phenotype Match", "Cluster Confidence", "Matching HPO terms"],
        index=0,
        key=sort_key,
    )

    active_ethnicity = st.session_state.get("eth_group_sel", "") or st.session_state.get("ethnicity", "")
    cnt_matrix = engine.ebl_count_matrix
    totals = engine.ebl_ethnicity_totals or {}

    def _compute_enrichment_ratio(gene_symbol: str) -> float | None:
        if not active_ethnicity or cnt_matrix is None or active_ethnicity not in cnt_matrix.columns:
            return None
        if active_ethnicity not in totals:
            return None
        y_val = totals.get(active_ethnicity, 0)
        if y_val <= 0:
            return None
        x_val = float(cnt_matrix.at[gene_symbol, active_ethnicity]) if gene_symbol in cnt_matrix.index else 0.0
        other_eths = [e for e in cnt_matrix.columns if e != active_ethnicity]
        y_other = sum(totals.get(e, 0) for e in other_eths)
        if y_other <= 0:
            return None
        x_other = sum(float(cnt_matrix.at[gene_symbol, e]) for e in other_eths if gene_symbol in cnt_matrix.index)
        p_target = x_val / y_val
        p_other = x_other / y_other
        if p_other <= 0:
            return float("inf") if p_target > 0 else None
        return p_target / p_other

    def _enrichment_marker(rr: float | None) -> tuple[str, str]:
        # Subtle 3-level palette for elevated enrichment only.
        if rr is None or rr < 2.0:
            return "", ""
        if rr == float("inf"):
            return "#b8473a", "High ethnic enrichment (present in selected ethnicity, absent in other groups)"
        if rr >= 7.0:
            return "#b8473a", f"High ethnic enrichment ({rr:.1f}× vs other groups)"
        if rr >= 4.0:
            return "#d1843f", f"Moderate ethnic enrichment ({rr:.1f}× vs other groups)"
        return "#d8b457", f"Mild ethnic enrichment ({rr:.1f}× vs other groups)"

    rows = []
    for gene in genes[:30]:
        stability = _stability_style(gene.stability)
        pct_match = round(len(gene.supporting_phenotypes) / max(len(observed_hpo_ids), 1) * 100) if observed_hpo_ids else 0
        rr = _compute_enrichment_ratio(gene.gene)
        marker_color, marker_tip = _enrichment_marker(rr)
        rows.append(
            {
                "obj": gene,
                "gene": gene.gene,
                "score": float(gene.score),
                "cluster": stability,
                "matching_hpo_terms": len(gene.supporting_phenotypes),
                "pct": pct_match,
                "best_match": gene.supporting_phenotypes[0] if gene.supporting_phenotypes else "-",
                "enrichment_rr": rr,
                "enrichment_marker_color": marker_color,
                "enrichment_marker_tip": marker_tip,
            }
        )

    if core_only:
        rows = [row for row in rows if row["obj"].stability.lower() == "core"]

    if sort_by == "Score":
        rows.sort(key=lambda row: row["score"], reverse=True)
    elif sort_by == "% Phenotype Match":
        rows.sort(key=lambda row: row["pct"], reverse=True)
    elif sort_by == "Matching HPO terms":
        rows.sort(key=lambda row: row["matching_hpo_terms"], reverse=True)
    else:
        order = {"High confidence": 0, "Moderate confidence": 1, "Low confidence": 2}
        rows.sort(key=lambda row: order.get(row["cluster"]["label"], 99))

    st.markdown(
        """
        <div class="card-shell" style="padding:0;overflow:hidden;">
          <div style="display:grid;grid-template-columns:52px 1.2fr 1.2fr 1.4fr .9fr 1.4fr;background:#f4f7fb;border-bottom:1px solid var(--border);padding:.75rem 1rem;">
            <div class="muted-table-head"></div>
            <div class="muted-table-head">Gene</div>
            <div class="muted-table-head">Score</div>
            <div class="muted-table-head">Cluster</div>
            <div class="muted-table-head">HPO Match</div>
            <div class="muted-table-head">Best Match</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, row in enumerate(rows):
        gene = row["obj"]
        is_open = st.session_state.get(state_key) == gene.gene
        row_cols = st.columns([0.55, 1.25, 1.35, 1.45, 0.95, 1.45])
        with row_cols[0]:
            if st.button("v" if is_open else ">", key=f"toggle_{table_key}_{gene.gene}_{idx}"):
                st.session_state[state_key] = None if is_open else gene.gene
                st.rerun()
        with row_cols[1]:
            sig_marker = ""
            if gene.supporting_phenotypes:
                best_id = _find_hpo_id_by_term(gene.supporting_phenotypes[0])
                sig_marker = " *" if engine.is_module_signature(best_id, st.session_state.get("current_top_module_id", -1)) else ""
            st.markdown(
                f'<div class="mono" style="font-size:14px;font-weight:700;color:var(--ink);padding-top:.45rem;">{_esc(gene.gene)}{sig_marker}</div>',
                unsafe_allow_html=True,
            )
        with row_cols[2]:
            marker_html = ""
            if row["enrichment_marker_color"]:
                marker_html = (
                    f'<span title="{_esc(row["enrichment_marker_tip"])}" '
                    f'style="display:inline-block;width:9px;height:9px;border-radius:999px;'
                    f'background:{row["enrichment_marker_color"]};box-shadow:0 0 0 1px rgba(15,23,42,.15);'
                    f'margin-left:.35rem;vertical-align:middle;"></span>'
                )
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:.55rem;padding-top:.55rem;">
                  <div style="width:70px;height:6px;border-radius:999px;background:#e8eef6;overflow:hidden;">
                    <div style="width:{max(min(gene.score, 1.0), 0.0) * 100:.1f}%;height:100%;background:linear-gradient(90deg,var(--blue),var(--teal));"></div>
                  </div>
                  <span class="mono" style="font-size:12px;color:var(--ink2);">{gene.score:.4f}</span>{marker_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with row_cols[3]:
            st.markdown(
                f'<div style="padding-top:.35rem;"><span style="font-size:12px;font-weight:600;padding:.28rem .6rem;border-radius:999px;background:{row["cluster"]["bg"]};color:{row["cluster"]["color"]};">{_esc(row["cluster"]["label"])}</span></div>',
                unsafe_allow_html=True,
            )
        with row_cols[4]:
            st.markdown(
                f'<div style="padding-top:.45rem;font-size:14px;font-weight:700;color:var(--ink);">{row["matching_hpo_terms"]} <span style="font-size:12px;font-weight:400;color:var(--ink3);">({row["pct"]}%)</span></div>',
                unsafe_allow_html=True,
            )
        with row_cols[5]:
            st.markdown(
                f'<div style="padding-top:.45rem;font-size:13px;color:var(--ink2);line-height:1.4;">{_esc(row["best_match"])}</div>',
                unsafe_allow_html=True,
            )
        if is_open:
            _render_gene_breakdown(gene)

# ─────────────────────────────────────────────────────────────────────────────
# UI Helper Components
# ─────────────────────────────────────────────────────────────────────────────

def _render_phenotype_chips(observed: list[str], excluded: list[str]) -> None:
    """Renders visual chips for selected phenotypes."""
    if not observed and not excluded:
        return

    html = '<div class="chip-container">'
    for term in observed:
        hid = _hpo_id(term)
        name = term.split("  (")[0]
        html += f'<div class="chip observed">✓ {name} ({hid})</div>'
    for term in excluded:
        hid = _hpo_id(term)
        name = term.split("  (")[0]
        html += f'<div class="chip excluded">✕ {name} ({hid})</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PDF builder (P3-B)
# ─────────────────────────────────────────────────────────────────────────────

def _build_pdf(result) -> bytes:
    """Build a clinical summary PDF and return as bytes. Requires reportlab."""
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Flowable
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.units import inch
        import datetime
    except ImportError:
        return b""

    class _LogoHeader(Flowable):
        """PDF header flowable — Stacked·Light-inspired logo mark + wordmark."""
        def __init__(self, width, gen_time, height=60):
            Flowable.__init__(self)
            self.width = width
            self.height = height
            self.gen_time = gen_time

        def draw(self):
            c = self.canv
            w, h = self.width, self.height
            teal  = rl_colors.HexColor("#009b8c")
            amber = rl_colors.HexColor("#d4820a")
            dark  = rl_colors.HexColor("#0e1620")

            # Background
            c.setFillColor(rl_colors.HexColor("#f0f4f8"))
            c.setStrokeColor(rl_colors.HexColor("#d6e4f0"))
            c.setLineWidth(0.5)
            c.roundRect(0, 0, w, h, 5, fill=1, stroke=1)

            # Icon — concentric rings centred at (ic_x, ic_y)
            ic_x, ic_y, sc = 38, h / 2, 0.295   # sc: SVG 120-unit space → ~35pt outer radius

            c.setStrokeColor(teal);  c.setLineWidth(0.5)
            c.circle(ic_x, ic_y, 52 * sc, fill=0, stroke=1)
            c.setLineWidth(0.7)
            c.circle(ic_x, ic_y, 38 * sc, fill=0, stroke=1)
            c.setStrokeColor(amber); c.setLineWidth(0.9)
            c.circle(ic_x, ic_y, 24 * sc, fill=0, stroke=1)

            def _node(sx, sy, col, r=1.3):
                c.setFillColor(col)
                c.circle(ic_x + (sx - 60) * sc, ic_y - (sy - 60) * sc, r, fill=1, stroke=0)

            _node(60, 22, teal)           # top
            _node(93, 41, teal, r=1.0)   # right-upper
            _node(93, 79, amber)          # right-lower
            _node(27, 79, amber, r=1.1)  # left-lower

            c.setFillColor(teal)
            c.circle(ic_x, ic_y, 6 * sc, fill=1, stroke=0)
            c.setFillColor(rl_colors.white)
            c.circle(ic_x, ic_y, 2.5 * sc, fill=1, stroke=0)

            # Divider
            div_x = ic_x + 52 * sc + 10
            c.setStrokeColor(teal); c.setLineWidth(0.4)
            c.line(div_x, 10, div_x, h - 10)

            # Wordmark
            tx = div_x + 10
            c.setFillColor(dark); c.setFont("Helvetica-Bold", 17)
            c.drawString(tx, ic_y + 5, "IRD")
            c.setFillColor(teal); c.setFont("Helvetica", 6.5)
            c.drawString(tx, ic_y - 9, "PRIORITIZATION ENGINE")

            # Right-side metadata
            c.setFillColor(rl_colors.HexColor("#8ca3bc"))
            c.setFont("Helvetica", 7.5)
            c.drawRightString(w - 8, ic_y + 5, "Clinical Summary Report")
            c.setFont("Helvetica", 7)
            c.drawRightString(w - 8, ic_y - 8, f"Generated: {self.gen_time}")

    doc_width = letter[0] - 2 * inch
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    body_style  = ParagraphStyle("body", parent=styles["Normal"], fontSize=10,
                                 fontName="Helvetica", spaceAfter=4, leading=14)
    head_style  = ParagraphStyle("head", parent=styles["Heading2"], fontSize=12,
                                 fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=12)

    story = []
    story.append(_LogoHeader(doc_width, datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")))
    story.append(Spacer(1, 0.25 * inch))

    # Top 3 module table
    story.append(Paragraph("Module Posterior Probabilities (Top 3)", head_style))
    mod_data = [["Module", "Clinical Label", "Probability"]]
    for mm in result.all_modules[:3]:
        mod_data.append([
            f"Module {mm.module_id}",
            MODULE_LABELS.get(mm.module_id, ""),
            f"{mm.probability * 100:.1f}%",
        ])
    mod_table = Table(mod_data, colWidths=[1.0 * inch, 3.5 * inch, 1.2 * inch])
    mod_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1a6b9a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#e8f2fb"), rl_colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#d6e4f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(mod_table)
    story.append(Spacer(1, 0.15 * inch))

    # Candidate genes (top 10)
    story.append(Paragraph("Candidate Genes (Top 10)", head_style))
    gene_data = [["Gene", "Score", "Cluster Confidence", "Matching HPO"]]
    for g in result.candidate_genes[:10]:
        gene_data.append([
            g.gene,
            f"{g.score:.4f}",
            _get_stability_icon(g.stability),
            str(len(g.supporting_phenotypes)),
        ])
    gene_table = Table(gene_data, colWidths=[1.2 * inch, 1.0 * inch, 1.5 * inch, 1.5 * inch])
    gene_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1a6b9a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#e8f2fb"), rl_colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#d6e4f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(gene_table)
    story.append(Spacer(1, 0.15 * inch))

    # Recommended Workup
    story.append(Paragraph("Recommended Workup", head_style))
    workup = result.phenotype_predictions.recommended_workup
    if workup:
        for t in workup[:15]:
            pct = f" — {t.prevalence * 100:.0f}%" if t.prevalence else ""
            story.append(Paragraph(f"🔬 <b>{t.term_name}</b> ({t.hpo_id}){pct}", body_style))
    else:
        story.append(Paragraph("No additional high-prevalence phenotypes to suggest.", body_style))
    story.append(Spacer(1, 0.10 * inch))

    # Prognostic Risk
    story.append(Paragraph("Prognostic Risk", head_style))
    risk = result.phenotype_predictions.prognostic_risk
    if risk:
        for t in risk[:15]:
            pct = f" — {t.prevalence * 100:.0f}%" if t.prevalence else ""
            story.append(Paragraph(f"📊 <b>{t.term_name}</b> ({t.hpo_id}){pct}", body_style))
    else:
        story.append(Paragraph("No prognostic risk phenotypes identified.", body_style))

    # Likely Next Manifestations
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph("Likely Next Manifestations", head_style))
    nxt = result.phenotype_predictions.likely_next_manifestations
    if nxt:
        for t in nxt:
            pct = f" — {t.prevalence * 100:.0f}%" if t.prevalence else ""
            story.append(Paragraph(
                f"🔭 <b>{t.term_name}</b> ({t.hpo_id}){pct}", body_style
            ))
    else:
        story.append(Paragraph("No likely next manifestations identified.", body_style))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Data: HPO 2026-04-13 · 442 IRD genes · 17 disease modules · MPV network analysis",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=7,
                       textColor=rl_colors.HexColor("#94a3b8")),
    ))

    doc.build(story)
    return buf.getvalue()


def _format_term_with_signature(hpo_id: str, term_name: str, top_module_id: int) -> str:
    """Add a star indicator if the term is a significant signature for the top module."""
    if engine.is_module_signature(hpo_id, top_module_id):
        return f"{term_name} ⭐"
    return term_name

# ─────────────────────────────────────────────────────────────────────────────
# Shared result renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_result(
    result,
    show_next_question: bool = True,
    observed_hpo_ids: list[str] | None = None,
    workup_add_to_query: bool = False,
) -> None:
    """Render a QueryResult — called by all three modes.

    observed_hpo_ids: HPO IDs used as the query's observed set (for % Phenotype Match).
    workup_add_to_query: when True (Phenotype Query only), workup/risk rows get an Add button.
    """

    def _truncate(s: str, max_len: int = 46) -> str:
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    tm = result.top_module
    n_genes = len(result.candidate_genes)
    obs = observed_hpo_ids or []
    n_obs = len(obs)

    mod_rows = [
        {"Module": f"Mod {mm.module_id}", "Probability": mm.probability * 100, "IsTop": mm.module_id == tm.module_id}
        for mm in sorted(result.all_modules, key=lambda m: m.module_id)
    ]
    chart_df = pd.DataFrame(mod_rows)
    base = alt.Chart(chart_df).encode(x=alt.X('Module:N', sort=None, title=""), y=alt.Y('Probability:Q', title='Probability (%)'))
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        color=alt.condition(alt.datum.IsTop, alt.value('#2a9d8f'), alt.value('#b8d0e8'))
    )
    rule = alt.Chart(pd.DataFrame({'y': [100/17]})).mark_rule(color='#e76f51', strokeDash=[4,4], size=2).encode(y='y')

    tab_ov, tab_genes, tab_clin = st.tabs(["Overview", "Candidate genes", "Workup and prognosis"])

    # --- Tab Persistence Hack (Option B) ---
    # Injects JS to click the correct tab after rerun if active_tab_idx > 0
    if st.session_state.active_tab_idx > 0:
        st.markdown(
            f"<script>setTimeout(function(){{"
            f"  var tabs = window.parent.document.querySelectorAll('[data-testid=\"stTab\"]');"
            f"  if (tabs.length > {st.session_state.active_tab_idx}) "
            f"    tabs[{st.session_state.active_tab_idx}].click();"
            f"}}, 200);</script>",
            unsafe_allow_html=True,
        )
        # Reset once used
        st.session_state.active_tab_idx = 0

    with tab_ov:
        st.subheader("Module Probability Distribution")
        st.markdown(
            '<p class="explain">Each bar shows the posterior probability that the patient '
            "belongs to that disease module given the input phenotypes. "
            "Taller bars = stronger evidence of that module. The red dashed line represents the prior probability (1/17).</p>",
            unsafe_allow_html=True,
        )

        col_chart, col_top3 = st.columns([3, 1])
        with col_chart:
            st.altair_chart((bars + rule).properties(height=260), use_container_width=True)

        with col_top3:
            st.markdown("**Top 3 modules**")
            for mm in result.all_modules[:3]:
                pct = mm.probability * 100
                label = _truncate(_module_label(mm.module_id))
                st.progress(int(min(pct, 100)) / 100, text=f"{label}: {pct:.1f}%")

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Top Module",
            _module_label(tm.module_id),
            help="The disease module with the highest Naive Bayes posterior probability given the observed phenotypes.",
        )
        m2.metric(
            "Module Probability",
            f"{tm.probability * 100:.1f}%",
            help="Posterior probability this patient's phenotype pattern matches the top module. Computed via Naive Bayes over 17 IRD disease clusters.",
        )
        m3.metric(
            "Confidence",
            f"{result.confidence * 100:.1f}%",
            help=(
                "Normalized entropy: 100% = the engine is certain (all probability mass on one module), "
                "0% = flat posterior (all 17 modules equally likely). Formula: 1 − H(posterior) / log₂(17)."
            ),
        )

        if tm.supporting_phenotypes:
            extras = tm.supporting_phenotypes
            display = ", ".join(extras[:6]) + ("…" if len(extras) > 6 else "")
            st.caption(f"Phenotypes supporting {_module_label(tm.module_id)}: {display}")

        nq_list = getattr(result, "next_questions", None) or (
            [result.next_question] if (hasattr(result, "next_question") and result.next_question) else []
        )
        if show_next_question and nq_list:
            st.divider()
            st.subheader("Next Suggested Questions (top 5)")
            st.markdown(
                '<p class="explain">Ranked by the same expected information gain as before: '
                "each suggestion is the next-best discriminating phenotype for reducing "
                "uncertainty in the module posterior (shown as a qualitative diagnostic value tier).</p>",
                unsafe_allow_html=True,
            )
            for qi, q in enumerate(nq_list[:5]):
                ig_label = _ig_display(q.information_gain)
                ig_tip   = _ig_tooltip(q.information_gain)
                st.markdown(
                    f'<div class="q-card">'
                    f'  <div class="q-name">{q.term_name}'
                    f'    <span class="q-id">{q.hpo_id}</span>'
                    f"  </div>"
                    f'  <div class="q-ig">Diagnostic value: {ig_label}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"ℹ️ {ig_tip}",
                    help=ig_tip,
                )

    with tab_genes:
        st.subheader(f"Candidate Genes ({n_genes})")
        st.markdown(
            '<p class="explain">Genes assigned to the top module, ranked by phenotypic '
            "overlap with the observed phenotypes. The score combines how well each gene's "
            "HPO profile matches the query (phenotype component) with a network stability "
            "bonus: core genes are rewarded, unstable genes are mildly penalised.</p>",
            unsafe_allow_html=True,
        )

        if result.candidate_genes:
            gene_rows = [
                {
                    "Gene": g.gene,
                    "Score": float(g.score),
                    "Cluster Confidence": _get_stability_icon(g.stability),
                    "Matching HPO terms": len(g.supporting_phenotypes),
                    "% Phenotype Match": (
                        round(len(g.supporting_phenotypes) / max(n_obs, 1) * 100)
                        if n_obs
                        else 0
                    ),
                    "Best matching phenotype": (
                        g.supporting_phenotypes[0] if g.supporting_phenotypes else "—"
                    ),
                    "NPP score": float(g.npp_score) if g.npp_score is not None else 0.0,
                }
                for g in result.candidate_genes[:30]
            ]

            filter_col, sort_col = st.columns([2, 2])
            with filter_col:
                core_only = st.checkbox(
                    "Show High cluster confidence genes only", value=False, key="filter_core"
                )
            with sort_col:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["Score", "% Phenotype Match", "Cluster Confidence", "Matching HPO terms", "NPP score"],
                    index=0,
                    key="gene_sort_by",
                )

            if core_only:
                gene_rows = [r for r in gene_rows if "High cluster confidence" in r["Cluster Confidence"]]

            reverse = True
            if sort_by == "Cluster Confidence":
                order = {
                    "🟢 High cluster confidence": 0,
                    "🟡 Moderate cluster confidence": 1,
                    "🔴 Low cluster confidence": 2,
                }
                gene_rows = sorted(gene_rows, key=lambda r: order.get(r["Cluster Confidence"], 99))
                reverse = False
            else:
                gene_rows = sorted(gene_rows, key=lambda r: r[sort_by], reverse=reverse)

            st.dataframe(
                pd.DataFrame(gene_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Gene":                  st.column_config.TextColumn("Gene", width="small"),
                    "Score":                 st.column_config.ProgressColumn(
                        "Score", format="%.4f", min_value=0, max_value=1.0,
                        help="Combined phenotype overlap score plus network stability bonus. "
                             "High cluster confidence genes receive a +0.2 bonus; Low cluster confidence genes a −0.2 penalty."
                    ),
                    "Cluster Confidence":    st.column_config.TextColumn("Cluster Confidence", width="medium"),
                    "Matching HPO terms":    st.column_config.NumberColumn("Matching HPO", width="small"),
                    "% Phenotype Match":     st.column_config.NumberColumn("% Phenotype Match", format="%d%%", width="small"),
                    "Best matching phenotype": st.column_config.TextColumn("Best match"),
                    "NPP score":             st.column_config.NumberColumn(
                        "NPP", format="%.4f", width="small",
                        help="Network Phenotype Priority score (reserved; not yet populated in this release)."
                    ),
                },
            )
            st.caption(
                "Cluster confidence reflects how consistently a gene co-clusters with its module across "
                "bootstrap iterations. Low confidence does not indicate a poor candidate — "
                "phenotype overlap score remains the primary ranking criterion."
            )

            # --- Technical Drill-down (Transparency Feature) ---
            st.markdown("---")
            st.markdown("### 🔍 Technical Drill-down")
            st.markdown(
                '<p class="explain">Select a gene to examine the specific mathematical '
                "contributions behind its rank.</p>",
                unsafe_allow_html=True
            )

            # --- Persist result for technical drill-down ---
            result = st.session_state.get("last_result")
            if result is None:
                st.info("Run a query first to see gene score breakdowns.")
                return

            # Map genes to their candidate objects for easy lookup
            gene_map = {g.gene: g for g in result.candidate_genes[:30]}
            selected_gene_name = st.selectbox(
                "Select gene to view score breakdown:",
                options=list(gene_map.keys()),
                key="gene_breakdown_selector"
            )

            if selected_gene_name:
                g = gene_map[selected_gene_name]
                with st.expander(f"Score breakdown — {g.gene}", expanded=False):
                    if not g.score_breakdown:
                        st.write("*No phenotype overlap with query — score is 0 for this gene.*")
                    else:
                        st.markdown("**Phenotype contributions:**")
                        # Show as a mini table
                        breakdown_data = [
                            {"Phenotype": name, "Contribution": f"+{contrib:.4f}"}
                            for name, contrib in g.score_breakdown
                        ]
                        st.table(breakdown_data)

                    if g.stability_breakdown:
                        cls, psi, mod = g.stability_breakdown
                        label = _get_stability_icon(cls).split(" ", 1)[1] # remove icon
                        st.markdown(f"**Stability ({label}, ψ={psi:.2f}):** module credit `+{mod:.4f}`")

                    if g.leak_breakdown:
                        st.markdown("**Module-extrapolated contributions (leak):**")
                        leak_data = [
                            {"Phenotype": name, "Imputed Credit": f"+{contrib:.4f}"}
                            for name, contrib in g.leak_breakdown
                        ]
                        st.table(leak_data)

                    st.markdown("---")
                    st.markdown(f"**Total score:** `{g.score:.4f}`")

            st.markdown("---")
            csv_bytes = pd.DataFrame(gene_rows).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download gene list (CSV)",
                data=csv_bytes,
                file_name="mpv_candidate_genes.csv",
                mime="text/csv",
                key="dl_genes_csv",
            )

            try:
                import reportlab  # noqa: F401 — check availability only
                pdf_bytes = _build_pdf(result)
                if pdf_bytes:
                    st.download_button(
                        label="⬇ Download clinical summary (PDF)",
                        data=pdf_bytes,
                        file_name="mpv_clinical_summary.pdf",
                        mime="application/pdf",
                        key="dl_summary_pdf",
                    )
            except ImportError:
                pass

        else:
            st.info("No genes are assigned to this module.")

    with tab_clin:
        col_workup, col_risk, col_next = st.columns(3)

        with col_workup:
            st.subheader("Recommended Workup")
            st.markdown(
                '<p class="explain">Phenotypes with prevalence ≥50% in the top module '
                "that have not yet been observed. These are likely present — actively "
                "looking for them is recommended.</p>",
                unsafe_allow_html=True,
            )
            workup = result.phenotype_predictions.recommended_workup
            if workup:
                for i, t in enumerate(workup[:15]):
                    pct = f" — {t.prevalence * 100:.0f}%" if t.prevalence else ""
                    display_name = _format_term_with_signature(t.hpo_id, t.term_name, result.top_module.module_id)
                    div = (
                        f'<div class="workup-item">🔬 <b>{display_name}</b> <code>{t.hpo_id}</code>{pct}</div>'
                    )
                    if workup_add_to_query:
                        row_l, row_r = st.columns([4, 1.2])
                        with row_l:
                            st.markdown(div, unsafe_allow_html=True)
                        with row_r:
                            if st.button(
                                "Add",
                                key=f"wu_add_{t.hpo_id}_{i}",
                                help="Add to observed phenotypes",
                                use_container_width=True,
                                on_click=_add_hpo_to_query_observed,
                                args=(t.hpo_id,)
                            ):
                                # Logic handled by on_click callback
                                pass
                    else:
                        st.markdown(div, unsafe_allow_html=True)
            else:
                st.markdown("*No additional high-prevalence phenotypes to suggest.*")

        with col_risk:
            st.subheader("Prognostic Risk")
            st.markdown(
                '<p class="explain">Phenotypes with prevalence 15–50% in the top module. '
                "Less certain than the workup list, but clinically meaningful — worth "
                "monitoring as the disease evolves.</p>",
                unsafe_allow_html=True,
            )
            risk = result.phenotype_predictions.prognostic_risk
            if risk:
                for i, t in enumerate(risk[:15]):
                    pct = f" — {t.prevalence * 100:.0f}%" if t.prevalence else ""
                    display_name = _format_term_with_signature(t.hpo_id, t.term_name, result.top_module.module_id)
                    div = (
                        f'<div class="risk-item">📊 <b>{display_name}</b> <code>{t.hpo_id}</code>{pct}</div>'
                    )
                    if workup_add_to_query:
                        row_l, row_r = st.columns([4, 1.2])
                        with row_l:
                            st.markdown(div, unsafe_allow_html=True)
                        with row_r:
                            if st.button(
                                "Add",
                                key=f"rk_add_{t.hpo_id}_{i}",
                                help="Add to observed phenotypes",
                                use_container_width=True,
                                on_click=_add_hpo_to_query_observed,
                                args=(t.hpo_id,)
                            ):
                                # Logic handled by on_click callback
                                pass
                    else:
                        st.markdown(div, unsafe_allow_html=True)
            else:
                st.markdown("*No prognostic risk phenotypes identified.*")

        with col_next:
            st.subheader("Likely Next Manifestations")
            st.markdown(
                '<p class="explain">Direct ontology children of observed phenotypes '
                "present in this module — the most probable next step in disease "
                "progression, surfaced regardless of prevalence threshold.</p>",
                unsafe_allow_html=True,
            )
            nxt = result.phenotype_predictions.likely_next_manifestations
            if nxt:
                for i, t in enumerate(nxt):
                    pct = f" — {t.prevalence * 100:.0f}%" if t.prevalence else ""
                    display_name = _format_term_with_signature(t.hpo_id, t.term_name, result.top_module.module_id)
                    div = (
                        f'<div class="next-item">🔭 <b>{display_name}</b> '
                        f'<code>{t.hpo_id}</code>{pct}</div>'
                    )
                    if workup_add_to_query:
                        row_l, row_r = st.columns([4, 1.2])
                        with row_l:
                            st.markdown(div, unsafe_allow_html=True)
                        with row_r:
                            if st.button(
                                "Add",
                                key=f"nx_add_{t.hpo_id}_{i}",
                                help="Add to observed phenotypes",
                                use_container_width=True,
                                on_click=_add_hpo_to_query_observed,
                                args=(t.hpo_id,)
                            ):
                                # Logic handled by on_click callback
                                pass
                    else:
                        st.markdown(div, unsafe_allow_html=True)
            else:
                st.markdown("*No progression phenotypes identified from observed terms.*")

    st.markdown(
        '<p style="font-size:0.75rem; color:#94a3b8; margin-top:1.5rem;">'
        "Data: HPO 2026-04-13 · 442 IRD genes · 17 disease modules · MPV network analysis"
        "</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — Phenotype Query
# ─────────────────────────────────────────────────────────────────────────────

def _query_mode() -> None:
    # 0. Drain the staging key before any widgets render
    # (Solves Widget vs Session State clobbering)
    if st.session_state.get("_query_observed_pending") is not None:
        st.session_state["query_observed"] = st.session_state.pop("_query_observed_pending")

    st.title("Phenotype Query")
    st.markdown(
        '<div class="intro-box">'
        "Enter phenotypes observed in your patient using the searchable dropdowns below. "
        "Type any part of the phenotype name or its HPO ID — the field filters as you type. "
        "Optionally add phenotypes that have been <em>explicitly ruled out</em> to the "
        "<strong>Excluded</strong> field — this sharpens diagnostic specificity. "
        "You can also bypass phenotype entry and query directly by gene symbol using the "
        "<strong>Gene-first query</strong> expander."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Demo Buttons ─────────────────────────────────────────────────────
    _render_clinical_cases()
    _render_real_clinical_cases()
    st.divider()

    # ── Input fields ──────────────────────────────────────────────────────
    col_obs, col_exc = st.columns(2)

    with col_obs:
        st.markdown(
            f"**Observed phenotypes ({len(st.session_state.get('query_observed', []))})** — "
            "confirmed present in this patient"
        )
        observed_fmt = st.multiselect(
            "observed",
            options=HPO_OPTIONS,
            placeholder="Type a phenotype name or HP: ID to search…",
            label_visibility="collapsed",
            key="query_observed",
        )

    with col_exc:
        st.markdown(
            f"**Excluded phenotypes ({len(st.session_state.get('query_excluded', []))})** — explicitly ruled out"
        )
        excluded_fmt = st.multiselect(
            "excluded",
            options=HPO_OPTIONS,
            placeholder="Type a phenotype name or HP: ID to search…",
            label_visibility="collapsed",
            key="query_excluded",
        )

    st.markdown("💡 " + _hpo_search_hint_md())

    # ── Gene-first expander ───────────────────────────────────────────────
    with st.expander("Gene-first query (optional)"):
        st.caption(
            "Query by gene instead of phenotype. The engine uses all HPO annotations "
            "for that gene as the 'observed' set and returns a result centred on its module."
        )
        gene_sel = st.selectbox(
            "Gene symbol",
            options=[""] + GENE_OPTIONS,
            index=0,
            format_func=lambda x: "\u2014 select a gene \u2014" if x == "" else x,
            key="query_gene",
        )

    # ── Render Visual Chips before running ────────────────────────────────
    _render_phenotype_chips(observed_fmt, excluded_fmt)

    # ── Run button ────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        run = st.button("Run Query", type="primary", use_container_width=True)

    # Trigger run if demo button OR 'Add' button was clicked
    if st.session_state.pop("run_demo", False) or st.session_state.pop("auto_run_query", False):
        run = True

    # Check for persisted result to handle widget reruns
    has_persisted = "last_result" in st.session_state

    if not run and not has_persisted:
        st.info(
            "Select observed and/or excluded phenotypes, or choose a gene under **Gene-first query**, "
            "then click **Run Query**."
        )
        return

    # ── Compute and display ───────────────────────────────────────────────
    if gene_sel:
        with st.spinner(f"Querying gene {gene_sel}…"):
            try:
                result = engine.query_gene(
                    gene_sel,
                    ethnicity_group=eth_group,
                    use_ethnicity_prior=use_eth_prior,
                )
            except ValueError as err:
                st.error(str(err))
                return
    else:
        observed = _hpo_ids(observed_fmt)
        excluded = _hpo_ids(excluded_fmt)
        if not observed and not excluded:
            st.warning("Please select at least one phenotype — or choose a gene in the expander.")
            return
        with st.spinner("Scoring disease modules…"):
            result = engine.query(
                observed,
                excluded,
                ethnicity_group=eth_group,
                use_ethnicity_prior=use_eth_prior,
            )

    if run:
        st.session_state["last_result"] = result
    else:
        result = st.session_state.get("last_result")

    st.success("Query complete — results below.")
    st.divider()
    if gene_sel:
        _render_result(
            result,
            observed_hpo_ids=engine.gene_observed_hpo_ids(gene_sel),
            workup_add_to_query=True,
        )
    else:
        _render_result(result, observed_hpo_ids=observed, workup_add_to_query=True)


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — Interactive Session
# ─────────────────────────────────────────────────────────────────────────────

def _session_mode() -> None:
    st.title("Interactive Session")
    st.markdown(
        '<div class="intro-box">'
        "The engine asks you one phenotype question at a time, always choosing the question "
        "whose answer would most sharpen the diagnostic distribution — as measured by "
        "information gain. After each answer the module posterior updates in real time. "
        "<br><br>"
        "Use <strong>Yes \u2014 Present</strong> / <strong>No \u2014 Absent</strong> / "
        "<strong>Skip</strong> to progress through questions, or add a phenotype you "
        "already know using the manual entry field. Click <strong>Reset</strong> to "
        "start a fresh session."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Session state ─────────────────────────────────────────────────────
    session_ctx = (gamma_val, eth_group, use_eth_prior)
    if (
        "sess_obj" not in st.session_state
        or st.session_state.get("sess_ctx") != session_ctx
    ):
        st.session_state.sess_obj      = engine.new_session(
            ethnicity_group=eth_group,
            use_ethnicity_prior=use_eth_prior,
        )
        st.session_state.sess_ctx      = session_ctx
        st.session_state.sess_history  = []  # list of (hpo_id, name, "yes"/"no"/"skip")
        st.session_state.sess_question = None
        st.session_state.sess_result   = None

    sess    = st.session_state.sess_obj
    history = st.session_state.sess_history

    if st.session_state.pop("_session_reset_ok", False):
        st.success("Session cleared — you can start fresh.")

    # ── Top action bar: manual add + reset ───────────────────────────────
    bar_left, bar_mid, bar_right = st.columns([3, 3, 1])

    with bar_left:
        manual_fmt = st.selectbox(
            "Manually add a known phenotype",
            options=[""] + HPO_OPTIONS,
            index=0,
            format_func=lambda x: "\u2014 search to add a known phenotype \u2014" if x == "" else x,
            key="sess_manual",
        )
        st.markdown("💡 " + _hpo_search_hint_md())
    with bar_mid:
        man_col_yes, man_col_no = st.columns(2)
        with man_col_yes:
            if st.button("Add as Present", use_container_width=True) and manual_fmt:
                hid  = _hpo_id(manual_fmt)
                name = engine.get_term_name(hid)
                sess.answer_yes(hid)
                history.append((hid, name, "yes"))
                st.session_state.sess_question = None
                st.session_state.sess_result   = None
                st.rerun()
        with man_col_no:
            if st.button("Add as Absent", use_container_width=True) and manual_fmt:
                hid  = _hpo_id(manual_fmt)
                name = engine.get_term_name(hid)
                sess.answer_no(hid)
                history.append((hid, name, "no"))
                st.session_state.sess_question = None
                st.session_state.sess_result   = None
                st.rerun()
    with bar_right:
        if st.button("Reset", use_container_width=True):
            sess.reset()
            st.session_state.sess_history  = []
            st.session_state.sess_question = None
            st.session_state.sess_result   = None
            st.session_state["_session_reset_ok"] = True
            st.rerun()

    st.divider()

    # ── Compute next question (cached until user answers) ─────────────────
    if st.session_state.sess_question is None:
        with st.spinner("Finding the most informative next question…"):
            q = sess.get_next_question()
            st.session_state.sess_question = q
            if sess.observed or sess.excluded:
                st.session_state.sess_result = sess.get_current_result()

    q = st.session_state.sess_question

    # ── Next question card ────────────────────────────────────────────────
    st.subheader("Current Question")
    st.markdown(
        '<p class="explain">The system chose this as the phenotype whose Yes/No answer '
        "would most reduce uncertainty about which disease module this patient belongs to. "
        "The diagnostic value tier reflects how much this question is expected to sharpen the diagnosis."
        "</p>",
        unsafe_allow_html=True,
    )
    _q_ig_label = _ig_display(q.information_gain)
    _q_ig_tip   = _ig_tooltip(q.information_gain)
    st.markdown(
        f'<div class="q-card">'
        f'  <div class="q-name">{q.term_name}'
        f'    <span class="q-id">{q.hpo_id}</span>'
        f"  </div>"
        f'  <div class="q-ig">Diagnostic value: {_q_ig_label}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"ℹ️ {_q_ig_tip}", help=_q_ig_tip)

    b_yes, b_no, b_skip = st.columns(3)
    with b_yes:
        if st.button("Yes \u2014 Present", type="primary", use_container_width=True):
            sess.answer_yes(q.hpo_id)
            history.append((q.hpo_id, q.term_name, "yes"))
            st.session_state.sess_question = None
            st.session_state.sess_result   = None
            st.rerun()
    with b_no:
        if st.button("No \u2014 Absent", use_container_width=True):
            sess.answer_no(q.hpo_id)
            history.append((q.hpo_id, q.term_name, "no"))
            st.session_state.sess_question = None
            st.session_state.sess_result   = None
            st.rerun()
    with b_skip:
        if st.button("Skip (unknown)", use_container_width=True):
            sess.excluded.append(q.hpo_id)
            history.append((q.hpo_id, q.term_name, "skip"))
            st.session_state.sess_question = None
            st.rerun()

    # ── Session history ───────────────────────────────────────────────────
    if history:
        st.divider()
        st.subheader(f"Answers Recorded ({len(history)})")
        st.markdown(
            '<p class="explain">Your answers so far, most recent first. '
            "Each answer was used to update the diagnostic posterior below.</p>",
            unsafe_allow_html=True,
        )
        icon   = {"yes": "✅", "no": "❌", "skip": "⏭️"}
        css    = {"yes": "hist-yes", "no": "hist-no", "skip": "hist-skip"}
        suffix = {"yes": "Present", "no": "Absent", "skip": "Skipped"}

        # P1-C + P2-C: Undo and Flip buttons for each history entry
        for rev_idx, (hpo_id, name, answer) in enumerate(reversed(history)):
            real_idx = len(history) - 1 - rev_idx
            row_left, row_undo, row_flip = st.columns([4, 1, 1])
            with row_left:
                st.markdown(
                    f'<div class="{css[answer]}">'
                    f'{icon[answer]} <b>{name}</b> '
                    f'<span style="color:#6a8eae">{hpo_id}</span> \u2014 {suffix[answer]}'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with row_undo:
                if st.button("↩ Undo", key=f"undo_{hpo_id}_{real_idx}"):
                    if answer == "yes" and hpo_id in sess.observed:
                        sess.observed.remove(hpo_id)
                    elif answer in ("no", "skip") and hpo_id in sess.excluded:
                        sess.excluded.remove(hpo_id)
                    del st.session_state.sess_history[real_idx]
                    st.session_state.sess_question = None
                    st.session_state.sess_result   = None
                    st.rerun()
            with row_flip:
                # Flip label: "→ No" if current answer is yes; "→ Yes" otherwise
                flip_label = "→ No" if answer == "yes" else "→ Yes"
                if st.button(flip_label, key=f"flip_{hpo_id}_{real_idx}"):
                    # Step 1: Undo old answer
                    if answer == "yes" and hpo_id in sess.observed:
                        sess.observed.remove(hpo_id)
                    elif answer in ("no", "skip") and hpo_id in sess.excluded:
                        sess.excluded.remove(hpo_id)
                    del st.session_state.sess_history[real_idx]
                    # Step 2: Apply opposite answer
                    if answer == "yes":
                        sess.answer_no(hpo_id)
                        new_answer = "no"
                    else:  # "no" or "skip"
                        sess.answer_yes(hpo_id)
                        new_answer = "yes"
                    # Step 3: Re-insert at same position
                    st.session_state.sess_history.insert(real_idx, (hpo_id, name, new_answer))
                    st.session_state.sess_question = None
                    st.session_state.sess_result   = None
                    st.rerun()

    # ── Current diagnosis ─────────────────────────────────────────────────
    if st.session_state.sess_result is not None:
        st.divider()
        st.subheader("Current Diagnosis")
        st.markdown(
            '<p class="explain">Module ranking and predictions based on all answers given so far. '
            "Updates automatically after each answer.</p>",
            unsafe_allow_html=True,
        )
        _render_result(
            st.session_state.sess_result,
            show_next_question=False,
            observed_hpo_ids=list(sess.observed),
        )


def _analytics_mode():
    """📊 Comparative Analytics: Dual-perspective dashboard for validation and research."""
    st.title("📊 Comparative Analytics")

    # ── Perspective Selector ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    perspective = st.segmented_control(
        "Choose Analysis Perspective",
        options=["Inter-Module Analysis", "Global Context (IRD vs. Universe)"],
        selection_mode="single",
        default="Inter-Module Analysis",
        label_visibility="collapsed",
        key="analytics_perspective"
    )

    st.divider()

    if perspective == "Inter-Module Analysis":
        # ── RetiGene Atlas Mapping ──────────────────────────────────────────────
        st.subheader("RetiGene Atlas Validation & Mapping")

        summary_path = "Input/modules_RetiGene_Comparison/enrichment_results_260412_1551.csv"
        dotplot_path  = "Input/modules_RetiGene_Comparison/enrichment_dotplot_260412_1551.png"

        if os.path.exists(summary_path):
            st.markdown("#### Enrichment Summary Table")
            try:
                df = pd.read_csv(summary_path)

                # Align new CSV columns to UI expectations
                rename_map = {
                    "module_id": "Module",
                    "category_group": "Category_group",
                    "value": "Value",
                    "overlap": "Overlap",
                    "fold_enrichment": "Fold_Enrichment",
                    "novel_candidates": "N_novel_candidates",
                    "potential_misclassifications": "N_potential_misclassifications"
                }
                df = df.rename(columns=rename_map)

                # Compute Recall and Precision if absent in new file
                if "Recall" not in df.columns:
                    df["Recall"] = (df["Overlap"] / df["N_category"]) * 100
                if "Precision" not in df.columns:
                    df["Precision"] = (df["Overlap"] / df["N_module"]) * 100

                # Clean percentage strings to numeric floats for styling (if any)
                for col in ["Recall", "Precision"]:
                    if df[col].dtype == object:
                        df[col] = df[col].str.rstrip('%').astype(float)

                # Map Module IDs to Clinical Labels
                df["Clinical Module"] = df["Module"].map(lambda m: _module_label(int(m)))

                # Reorder columns to put Clinical Module first
                cols = ["Clinical Module"] + [c for c in df.columns if c not in ["Module", "Clinical Module"]]
                df = df[cols]

                st.dataframe(
                    df.style.background_gradient(subset=["Recall", "Precision"], cmap="Reds", vmin=0, vmax=100),
                    use_container_width=True,
                    height=500,
                    hide_index=True,
                    column_config={
                        "Clinical Module":       st.column_config.TextColumn("Clinical Module", width="medium"),
                        "Category_group":        st.column_config.TextColumn("Analysis Category", width="small"),
                        "Value":                 st.column_config.TextColumn("Atlas Annotation"),
                        "Overlap":               st.column_config.NumberColumn("Co-annotated Genes", width="small"),
                        "N_category":            st.column_config.NumberColumn("Atlas Group Size", width="small"),
                        "Recall":                st.column_config.NumberColumn("Recovery Rate (Recall)", format="%.1f%%", width="small"),
                        "Precision":             st.column_config.NumberColumn("Module Specificity (Precision)", format="%.1f%%", width="small"),
                        "Fold_Enrichment":       st.column_config.NumberColumn("Enrichment Fold", format="%.2f", width="small"),
                        "FDR":                   st.column_config.NumberColumn("Significance (p-adj)", format="%.2e", width="small"),
                        "N_novel_candidates":    st.column_config.NumberColumn("Novel candidates", width="small"),
                        "N_potential_misclassifications": st.column_config.NumberColumn("Mapping Divergence", width="small"),
                        "N_module":              st.column_config.NumberColumn("Mod Size", width="small"),
                    }
                )
                st.markdown(
                    '<p class="explain">This table details 60 statistically significant functional and phenotypic enrichments '
                    "identified across all 17 gene modules, applying a Benjamini-Hochberg False Discovery Rate (FDR) "
                    "threshold of < 0.05. It utilizes <b>Recall</b> and <b>Precision</b> to quantify the alignment "
                    "between the HPO-based clusters and curated RetiGene categories, while <b>Fold Enrichment</b> "
                    "represents the ratio of observed to expected gene overlap. Additionally, the identification of "
                    "<b>Novel Candidates</b> highlights genes within a module that are currently unannotated for a "
                    "specific category—suggesting potential new disease associations—whereas "
                    "<b>Potential Misclassifications</b> list known category genes that were not captured by the module.</p>",
                    unsafe_allow_html=True
                )
                st.caption("Detailed record of significant hit associations.")
            except Exception as e:
                st.error(f"Error loading RetiGene summary: {e}")
        else:
            st.warning("RetiGene enrichment summary file not found.")

        st.divider()

        # ── Dotplot Visualization ─────────────────────────────────────────────
        if os.path.exists(dotplot_path):
            st.markdown("#### Global Enrichment Dotplot")
            st.image(
                dotplot_path,
                caption="Visual summary of module-category associations.",
                use_container_width=True
            )
            st.markdown(
                '<p class="explain">This dot plot summarizes the distribution and statistical strength of significant '
                "functional and phenotypic enrichments across the 17 gene modules. Each point represents a "
                "statistically significant association between a module and a specific category value from the "
                "RetiGene database. The <b>size of the dot</b> is proportional to the <b>Fold Enrichment</b>, "
                "reflecting the magnitude of over-representation, while the <b>color intensity</b> indicates the "
                "statistical significance, mapped to <b>-log<sub>10</sub>(FDR)</b>. The visualization is faceted by "
                "category group to allow for comparative analysis across disease subtypes, functional pathways, "
                "and phenotypic classifications.</p>",
                unsafe_allow_html=True
            )
        else:
            st.warning("RetiGene enrichment dotplot image not found.")

        st.divider()

        # ── Comparative Signatures ─────────────────────────────────────────────
        st.subheader("Comparative Signatures (Cross-Module)")

        comp_sigs_pdf = "Input/comparative_signatures_all_IRD/comparative_signatures_20260413_0959.pdf"

        if os.path.exists(comp_sigs_pdf):
            st.markdown(
                '<div class="intro-box" style="background:#f8fafc; border-left-color:#94a3b8;">'
                "The full Comparative Signature Matrix is available as a high-resolution PDF "
                "for detailed cross-module analysis."
                "</div>",
                unsafe_allow_html=True
            )
            with open(comp_sigs_pdf, "rb") as f:
                pdf_data = f.read()
            st.download_button(
                label="⬇ Download Comparison Matrix (High-Resolution PDF)",
                data=pdf_data,
                file_name="comparative_signatures_matrix.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="dl_matrix_analytics"
            )
        else:
            st.warning("Comparative signatures PDF not found.")

        st.markdown(
            """
            ### **Module-Specific Phenotypic Signatures: Comparative Multi-Panel Overview**

            This multi-panel bar chart provides a side-by-side comparative analysis of the unique phenotypic signatures defining individual IRD gene modules. By visualizing the prevalence of 299 phenotypes across 17 modules, the figure illustrates how specific clinical traits are partitioned within the genotype-phenotype architecture of Inherited Retinal Diseases.

            | Aspect | Details |
            | :--- | :--- |
            | **Visualization** | Faceted multi-panel bar chart |
            | **Comparison** | Module-specific profiles (Modules 0–16) compared against the global IRD background (`All_IRD`) |
            | **Categorization** | Phenotypes are grouped by physiological systems (e.g., Brain Morphology, Eye Movement, Renal Morphology, Peripheral Neuropathy) |

            ---

            ### **Description**

            *   **Architecture**: The visualization is organized into vertical columns representing individual gene modules, allowing for direct horizontal comparison of phenotype prevalence. The `All_IRD` column serves as the global baseline for the entire cohort.
            *   **Systemic Grouping**: Phenotypes are hierarchically grouped into high-level clinical categories (e.g., *Abnormal renal morphology*, *Peripheral neuropathy*). This structure highlights which modules exhibit extra-ocular (syndromic) manifestations versus those restricted to primary retinal traits.
            *   **Signature Quantification**: Each horizontal bar represents the prevalence or statistical frequency of a specific **Human Phenotype Ontology (HPO)** term within a module. The presence of a colored bar indicates that the phenotype is a component of that module’s signature.
            *   **Defining Features**: Specific markers (black dots) denote module-defining traits where the association is particularly strong or statistically significant. For example, **Module 4** displays a distinct signature for *Abnormal brain morphology* (including the "Molar tooth sign"), while **Module 7** is characterized by *Abnormal renal morphology*.
            *   **Clinical Utility**: This overview enables the rapid identification of "clinical fingerprints" for each genetic cluster, facilitating the differentiation between non-syndromic modules and complex syndromic clusters (such as those associated with ciliopathies or mitochondrial dysfunction).
            """
        )

    else:
        # ── Global Context (IRD vs. Universe) ──────────────────────────────────
        st.subheader("Global Context: IRD vs. Gene Universe")
        st.markdown(
            '<div class="intro-box">'
            "Analysis of Inherited Retinal Diseases (IRD) in the context of the global gene universe. "
            "This table focuses on identifying phenotypes that are significantly over-represented "
            "in IRD genes compared to the background of all known human genes."
            "</div>",
            unsafe_allow_html=True
        )

        global_csv_path = "Input/IRD_vs_Non-IRD/IRD_vs_Non-IRD.csv"

        if os.path.exists(global_csv_path):
            try:
                df_global = pd.read_csv(global_csv_path)

                # Cleanup and Format
                # Rename columns for clarity
                column_mapping = {
                    "hpo_name":      "Phenotype Name",
                    "target_count":  "IRD Gene Count",
                    "bg_count":      "Global Gene Count",
                    "pct_target":    "Prevalence in IRD (%)",
                    "pct_bg":        "Prevalence Globally (%)",
                    "odds_ratio":    "Enrichment Magnitude (OR)",
                    "q_value":       "Significance (FDR)",
                    "IC":            "Info Content",
                    "depth":         "HPO Depth",
                    "HPO_ID":        "HPO ID"
                }

                # Filter useful columns and rename
                df_disp = df_global[list(column_mapping.keys())].rename(columns=column_mapping)

                # Handle infinite values (often caused by 0 global background count)
                # We replace infinity with a high finite value so the color scale works
                df_disp["Enrichment Magnitude (OR)"] = df_disp["Enrichment Magnitude (OR)"].replace([np.inf, -np.inf], 100.0)

                # Convert fractions to absolute percentages
                df_disp["Prevalence in IRD (%)"]   = df_disp["Prevalence in IRD (%)"] * 100
                df_disp["Prevalence Globally (%)"] = df_disp["Prevalence Globally (%)"] * 100

                # Sort by Odds Ratio descending to show most enriched phenotypes first
                df_disp = df_disp.sort_values("Enrichment Magnitude (OR)", ascending=False)

                # Filter for Top 25% based on Odds Ratio
                threshold = df_disp["Enrichment Magnitude (OR)"].quantile(0.75)
                df_disp = df_disp[df_disp["Enrichment Magnitude (OR)"] >= threshold]

                st.dataframe(
                    df_disp.style.background_gradient(subset=["Enrichment Magnitude (OR)"], cmap="Reds"),
                    use_container_width=True,
                    height=600,
                    hide_index=True,
                    column_config={
                        "Prevalence in IRD (%)":   st.column_config.NumberColumn(format="%.1f%%"),
                        "Prevalence Globally (%)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Enrichment Magnitude (OR)": st.column_config.NumberColumn(format="%.2f"),
                        "Significance (FDR)":      st.column_config.NumberColumn(format="%.2e"),
                        "Info Content":            st.column_config.NumberColumn(format="%.2f"),
                    }
                )

                st.info(f"Table is filtered to show **Top 25%** most enriched phenotypes (OR ≥ {threshold:.2f}).")

                st.markdown(
                    '<p class="explain">This global enrichment table highlights HPO terms that are '
                    "distinctively characteristic of IRD genes. The <b>Enrichment Magnitude (Odds Ratio)</b> "
                    "measures how much more likely a phenotype is to be associated with an IRD gene "
                    "than a random gene in the genome. Phenotypes with very high OR values (highlighted in red) "
                    "represent the unique clinical identifiers of the retinal disease landscape.</p>",
                    unsafe_allow_html=True
                )

                st.divider()

                # ── Global Signature Barplot Visualization ────────────────────────────
                barplot_path = "Input/IRD_vs_Non-IRD/signature_barplot_ird_universe_LIST.png"
                if os.path.exists(barplot_path):
                    st.markdown("#### IRD Global Phenotypic Signature")
                    st.image(
                        barplot_path,
                        caption="Comparative visualization of phenotypic prevalence: IRD Cohort vs. Global Gene Universe background.",
                        use_container_width=True
                    )
                    st.markdown(
                        '<p class="explain">This phenotypic signature plot provides a comprehensive '
                        "overview of the global clinical profile of the IRD Universe (452 genes) "
                        "compared against a non-IRD background. This version displays a curated list "
                        "of 97 phenotypic clusters—including inheritance patterns and systemic clinical traits—derived "
                        "from MICA-based semantic clustering. The visualization contrasts phenotype prevalence "
                        "between the background (left) and the IRD cohort (right), with statistical significance "
                        "determined by Fisher’s exact tests and Benjamini-Hochberg FDR correction. By presenting "
                        "this expanded selection of phenotypes, the analysis identifies the primary clinical "
                        "hallmarks and hereditary architectures that define the broader IRD landscape.</p>",
                        unsafe_allow_html=True
                    )

                st.divider()

                # Download button for global analysis
                csv_bytes_global = df_disp.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download Global Enrichment Data (CSV)",
                    data=csv_bytes_global,
                    file_name="ird_vs_global_enrichment.csv",
                    mime="text/csv",
                    key="dl_global_analytics"
                )

            except Exception as e:
                st.error(f"Error loading global context data: {e}")
        else:
            st.warning("Global enrichment data file not found at Input/IRD_vs_Non-IRD/IRD_vs_Non-IRD.csv")

        st.divider()
        st.caption("Benchmarking IRD architecture against the human genome background.")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — Module Browser
# ─────────────────────────────────────────────────────────────────────────────

def _browser_mode() -> None:
    st.title("Module Browser")
    st.markdown(
        '<div class="intro-box">'
        "Browse the 17 IRD disease modules identified by network analysis of "
        "gene\u2013phenotype associations. Each module clusters genes and diseases "
        "that are genetically and phenotypically related. "
        "Select a module to explore its HPO phenotype profile and the genes assigned to it."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Module selector + stats ────────────────────────────────────────────
    sel_col, stat1, stat2 = st.columns([2, 1, 1])

    with sel_col:
        module_id = st.selectbox(
            "Select a disease module",
            options=list(range(17)),
            format_func=lambda m: _module_label(m),
        )

    with st.spinner(f"Loading Module {module_id}\u2026"):
        data = engine.browse_module(module_id)

    with stat1:
        st.metric("Genes in module", data["gene_count"], help="Number of IRD genes assigned to this module by network clustering")
    with stat2:
        st.metric("Annotated HPO terms", data["annotated_term_count"], help="HPO terms with prevalence > 0 in this module")

    # ── Tabs for terms, genes, and system plot ────────────────────────────
    tab_terms, tab_genes, tab_system = st.tabs(
        [
            f"HPO Phenotype Profile ({data['annotated_term_count']})",
            f"Gene List ({data['gene_count']})",
            "System Architecture",
        ]
    )

    with tab_terms:
        st.markdown(
            '<p class="explain">All HPO phenotypes with non-zero prevalence in this module, '
            "sorted from most to least common.</p>",
            unsafe_allow_html=True,
        )
        if data["top_terms"]:
            st.markdown("**Prevalence-based profile** (same source as before)")
            df_terms = pd.DataFrame(data["top_terms"])
            st.dataframe(
                df_terms,
                use_container_width=True,
                height=400,
                hide_index=True,
                column_config={
                    "HPO ID":           st.column_config.TextColumn("HPO ID", width="small"),
                    "Term":             st.column_config.TextColumn("Phenotype name"),
                    "Module prevalence":st.column_config.TextColumn("Prevalence in module", width="small"),
                },
            )
        else:
            st.info("No annotated HPO terms found for this module.")

        st.divider()
        st.markdown(
            f"**🔑 Module Signatures** ({data['signature_count']} terms) — "
            "Clinically discriminating phenotypes for this module (FDR corrected)."
        )
        st.markdown(
            '<p class="explain">These are phenotypes that are statistically enriched in this module '
            "compared to all other IRD modules. A high Odds Ratio indicates strong diagnostic specificity.</p>",
            unsafe_allow_html=True,
        )
        if data["signatures"]:
            df_sigs = pd.DataFrame(data["signatures"])
            st.dataframe(
                df_sigs,
                use_container_width=True,
                height=400,
                hide_index=True,
                column_config={
                    "HPO ID":           st.column_config.TextColumn("HPO ID", width="small"),
                    "Term":             st.column_config.TextColumn("Phenotype name"),
                    "Odds Ratio":       st.column_config.NumberColumn("Odds Ratio", format="%.1f", width="small"),
                    "q-value":          st.column_config.TextColumn("Significance (q)", width="small"),
                    "Freq in Module":   st.column_config.TextColumn("Freq", width="small"),
                    "Specificity Ratio":st.column_config.NumberColumn("Specificity", format="%.1f", width="small"),
                },
            )
            st.caption("Signatures are identified by high Odds Ratio and FDR significance (q-value < 0.05).")

            # Cross-panel navigation button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("View full cross-module comparison analytics", type="primary", use_container_width=True):
                st.session_state.app_mode = "📊 Comparative Analytics"
                st.rerun()
        else:
            st.info("No significant module-specific signatures identified.")

    with tab_genes:
        st.markdown(
            '<p class="explain">All genes assigned to this module by network clustering, '
            "ordered by stability score. <b>High cluster confidence</b> genes are tightly and "
            "consistently clustered; <b>Moderate cluster confidence</b> genes have intermediate "
            "evidence; <b>Low cluster confidence</b> genes have uncertain assignment and may "
            "overlap other modules. "
            "The HPO annotations column shows how many phenotypes are documented for that gene.</p>",
            unsafe_allow_html=True,
        )
        if data["genes"]:
            display_genes = [
                {
                    **g,
                    "Cluster Confidence": _get_stability_icon(g.get("Stability", ""))
                }
                for g in data["genes"]
            ]

            # Filter and sort controls for browser gene table
            filter_col_b, sort_col_b = st.columns([2, 2])
            with filter_col_b:
                core_only_b = st.checkbox(
                    "Show High cluster confidence genes only", value=False, key="filter_core_browser"
                )
            with sort_col_b:
                sort_by_b = st.selectbox(
                    "Sort by",
                    options=["Stability score", "Cluster Confidence", "HPO annotations"],
                    index=0,
                    key="gene_sort_by_browser",
                )

            if core_only_b:
                display_genes = [g for g in display_genes if "High cluster confidence" in g.get("Cluster Confidence", "")]

            if sort_by_b == "Cluster Confidence":
                order_b = {
                    "🟢 High cluster confidence": 0,
                    "🟡 Moderate cluster confidence": 1,
                    "🔴 Low cluster confidence": 2,
                }
                display_genes = sorted(display_genes, key=lambda g: order_b.get(g.get("Cluster Confidence", ""), 99))
            else:
                display_genes = sorted(display_genes, key=lambda g: g.get(sort_by_b, 0), reverse=True)

            st.dataframe(
                pd.DataFrame(display_genes),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Gene":               st.column_config.TextColumn("Gene", width="small"),
                    "Cluster Confidence": st.column_config.TextColumn("Cluster Confidence", width="medium"),
                    "Stability score":    st.column_config.NumberColumn("Stability score", format="%.4f", width="small"),
                    "HPO annotations":    st.column_config.NumberColumn("HPO annotations", width="small"),
                },
            )
            st.caption(
                "Cluster confidence reflects how consistently a gene co-clusters with its module across "
                "bootstrap iterations. Low confidence does not indicate a poor candidate — "
                "phenotype overlap score remains the primary ranking criterion."
            )

            # P3-A: CSV export for module gene list
            csv_bytes_mod = pd.DataFrame(display_genes).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download module gene list (CSV)",
                data=csv_bytes_mod,
                file_name=f"mpv_module_{module_id}_genes.csv",
                mime="text/csv",
                key=f"dl_mod_genes_{module_id}",
            )
        else:
            st.info("No genes assigned to this module.")

    with tab_system:
        _render_explain(
            "Network-level visualisation of the selected module. Each node represents an IRD gene; "
            "edges reflect shared phenotypic and protein-interaction evidence used during clustering. "
            "Node colour encodes cluster confidence (core / peripheral / unstable); "
            "node size scales with the number of annotated HPO terms."
        )
        system_plot_path = f"Input/single_system_plots/system_plot_module_{module_id}.png"
        if os.path.exists(system_plot_path):
            st.image(
                system_plot_path,
                caption=f"System architecture — Module {module_id}: {_module_label(module_id)}",
                use_container_width=True,
            )
        else:
            st.info(f"No system plot available for Module {module_id}.")

        st.divider()
        _render_label("All-Module Overview")
        _render_explain(
            "Global view of all IRD modules in the network landscape. "
            "Use this to orient the selected module within the broader IRD disease space."
        )
        all_plot_path = "Input/single_system_plots/system_plot_module_All.png"
        if os.path.exists(all_plot_path):
            st.image(
                all_plot_path,
                caption="Full IRD gene network — all modules shown simultaneously. "
                        "Colour encodes module identity; layout reflects phenotypic and network proximity.",
                use_container_width=True,
            )


def _render_result(
    result,
    show_next_question: bool = True,
    observed_hpo_ids: list[str] | None = None,
    workup_add_to_query: bool = False,
) -> None:
    tm = result.top_module
    st.session_state["current_top_module_id"] = tm.module_id
    observed_hpo_ids = observed_hpo_ids or []

    tab_overview, tab_genes, tab_clin = st.tabs(["Overview", "Candidate genes", "Workup and prognosis"])

    if st.session_state.active_tab_idx > 0:
        st.markdown(
            f"<script>setTimeout(function(){{"
            f"var tabs = window.parent.document.querySelectorAll('[data-testid=\"stTab\"]');"
            f"if (tabs.length > {st.session_state.active_tab_idx}) tabs[{st.session_state.active_tab_idx}].click();"
            f"}}, 200);</script>",
            unsafe_allow_html=True,
        )
        st.session_state.active_tab_idx = 0

    with tab_overview:
        hero_col, gauge_col = st.columns([2.2, 1])
        with hero_col:
            _render_top_module_hero(result)
        with gauge_col:
            _render_conf_gauge(result.confidence)

        top_three = result.all_modules[:3]
        summary_cols = st.columns(3)
        for idx, mm in enumerate(top_three):
            with summary_cols[idx]:
                st.markdown(
                    f"""
                    <div class="card-shell" style="padding:1rem;height:100%;">
                      <div class="label">Rank {idx + 1}</div>
                      <div style="font-family:Outfit,sans-serif;font-size:1.1rem;font-weight:800;color:{'var(--teal)' if idx == 0 else 'var(--ink)'};margin-top:.35rem;">Module {mm.module_id}</div>
                      <div style="font-size:13px;line-height:1.45;color:var(--ink2);margin-top:.25rem;">{_esc(MODULE_LABELS.get(mm.module_id, 'Unknown'))}</div>
                      <div class="mono" style="font-size:13px;font-weight:700;color:{'var(--teal)' if idx == 0 else 'var(--ink3)'};margin-top:.55rem;">{mm.probability * 100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        _render_explain(
            "All 17 modules are sorted by posterior probability. The top module is highlighted and the red marker shows the uniform prior."
        )
        _render_module_chart(result, title="Module Posterior Distribution")

        next_questions = getattr(result, "next_questions", None) or (
            [result.next_question] if getattr(result, "next_question", None) else []
        )
        if show_next_question and next_questions:
            st.markdown("<div style='height:.6rem;'></div>", unsafe_allow_html=True)
            _render_label("Next Suggested Questions")
            _render_explain(
                "Questions are ranked by expected information gain. Each bar shows the fraction of the maximum entropy reduction available across 17 modules."
            )
            _render_next_questions(next_questions[:5])

    with tab_genes:
        _render_label(f"Candidate Genes ({len(result.candidate_genes)})")
        _render_explain(
            "Gene ranking combines phenotype overlap with the query and the existing cluster stability modifier. Expand a row to inspect phenotype contributions and the stability term."
        )
        if result.candidate_genes:
            _render_gene_table(result.candidate_genes, observed_hpo_ids, table_key="candidate")
            export_rows = []
            for gene in result.candidate_genes[:30]:
                export_rows.append(
                    {
                        "Gene": gene.gene,
                        "Score": round(float(gene.score), 4),
                        "Cluster Confidence": _stability_style(gene.stability)["label"],
                        "Matching HPO terms": len(gene.supporting_phenotypes),
                        "% Phenotype Match": (
                            round(len(gene.supporting_phenotypes) / max(len(observed_hpo_ids), 1) * 100)
                            if observed_hpo_ids
                            else 0
                        ),
                        "Best matching phenotype": gene.supporting_phenotypes[0] if gene.supporting_phenotypes else "-",
                    }
                )
            csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
            download_cols = st.columns([1, 1, 2])
            with download_cols[0]:
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name="mpv_candidate_genes.csv",
                    mime="text/csv",
                    key="dl_genes_csv_redesign",
                    use_container_width=True,
                )
            with download_cols[1]:
                try:
                    import reportlab  # noqa: F401

                    pdf_bytes = _build_pdf(result)
                    if pdf_bytes:
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name="mpv_clinical_summary.pdf",
                            mime="application/pdf",
                            key="dl_summary_pdf_redesign",
                            use_container_width=True,
                        )
                except ImportError:
                    pass
        else:
            st.info("No genes are assigned to this module.")

    with tab_clin:
        clinical_cols = st.columns(3)
        with clinical_cols[0]:
            _render_workup_column(
                title="Recommended Workup",
                subtitle="Prevalence >= 50% in the top module. These are the highest-priority findings to actively look for next.",
                items=result.phenotype_predictions.recommended_workup[:15],
                accent_color="var(--blue)",
                accent_bg="var(--blue-l)",
                accent_border="#bfdbfe",
                icon="Workup",
                add_prefix="wu_add_redesign" if workup_add_to_query else None,
            )
        with clinical_cols[1]:
            _render_workup_column(
                title="Prognostic Risk",
                subtitle="Prevalence 15-50% in the top module. Lower certainty than workup items, but clinically meaningful to monitor.",
                items=result.phenotype_predictions.prognostic_risk[:15],
                accent_color="var(--amber)",
                accent_bg="var(--amber-l)",
                accent_border="#fcd34d",
                icon="Risk",
                add_prefix="rk_add_redesign" if workup_add_to_query else None,
            )
        with clinical_cols[2]:
            _render_workup_column(
                title="Likely Next Manifestations",
                subtitle="Ontology children of currently observed phenotypes that are plausible next progression steps in this module.",
                items=result.phenotype_predictions.likely_next_manifestations,
                accent_color="var(--emerald)",
                accent_bg="var(--emerald-l)",
                accent_border="#6ee7b7",
                icon="Next",
                add_prefix="nx_add_redesign" if workup_add_to_query else None,
            )

    st.markdown(
        '<div class="footer-meta mono">MPV Phenotype Engine · Naive Bayes · 17 modules · 442 IRD genes · HPO 2026-04-13</div>',
        unsafe_allow_html=True,
    )


def _query_mode() -> None:
    if st.session_state.get("_query_observed_pending") is not None:
        st.session_state["query_observed"] = st.session_state.pop("_query_observed_pending")

    _render_page_header(
        "Phenotype Query",
        "This tool compares a patient's clinical features, written as HPO terms, against inherited retinal disease modules.\n\n"
        "It estimates the most likely disease module and prioritizes candidate genes for review.",
    )
    _render_case_load_notice()

    st.divider()

    _render_section_header(
        "Start From a Case",
        "Use an example instead of typing phenotypes manually",
        "These presets show how the engine behaves with known phenotype patterns. They are useful for testing the workflow or demonstrating typical IRD presentations.",
    )
    _render_clinical_cases()
    _render_real_clinical_cases()

    st.divider()

    st.markdown(
        """
        <div class="section-title">Phenotype Selection</div>
        <div class="section-subtitle">Describe the patient's observed and excluded clinical findings</div>
        <p class="section-copy">
          <span style="color:var(--teal);font-weight:550;">Observed</span> phenotypes support a diagnosis.<br>
          <span style="color:var(--red);font-weight:550;">Excluded</span> phenotypes help the engine reduce support for modules where those findings are expected.
        </p>
        """,
        unsafe_allow_html=True,
    )
    selector_cols = st.columns(2)
    with selector_cols[0]:
        _render_label(f"Observed Phenotypes ({len(st.session_state.get('query_observed', []))})")
        observed_fmt = st.multiselect(
            "Observed phenotypes",
            options=HPO_OPTIONS,
            placeholder="Type a phenotype name or HP: ID to search...",
            label_visibility="collapsed",
            key="query_observed",
        )
    with selector_cols[1]:
        _render_label(f"Excluded Phenotypes ({len(st.session_state.get('query_excluded', []))})")
        excluded_fmt = st.multiselect(
            "Excluded phenotypes",
            options=HPO_OPTIONS,
            placeholder="Type a phenotype name or HP: ID to search...",
            label_visibility="collapsed",
            key="query_excluded",
        )

    st.markdown(
        '<p class="soft-note"><strong>Search tip:</strong> use an HPO name or HP ID. Full ontology: '
        '<a href="https://hpo.jax.org/" target="_blank">hpo.jax.org</a>.</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    _render_section_header(
        "Gene-First Query",
        "Start from a gene when a candidate gene is already known",
        "The engine uses that gene's annotated HPO profile as the observed phenotype set.",
    )

    with st.expander("Gene-first query", expanded=False):
        gene_sel = st.selectbox(
            "Gene symbol",
            options=[""] + GENE_OPTIONS,
            index=0,
            format_func=lambda value: "-- select a gene --" if value == "" else value,
            key="query_gene",
        )

    st.divider()

    _render_section_header(
        "Review and Run",
        "Confirm the query before scoring",
        "Selected terms appear below as chips. Run the engine when the profile is ready.",
    )
    _render_phenotype_chips(observed_fmt, excluded_fmt)

    run_cols = st.columns([2, 1, 2])
    with run_cols[1]:
        run = st.button("Run Query", type="primary", use_container_width=True)

    if st.session_state.pop("run_demo", False) or st.session_state.pop("auto_run_query", False):
        run = True

    result = None
    observed = _hpo_ids(observed_fmt)
    excluded = _hpo_ids(excluded_fmt)

    if not run and "last_result" not in st.session_state:
        st.info("Select phenotypes or choose a gene, then run the engine.")
        return

    if run:
        if gene_sel:
            with st.spinner(f"Querying gene {gene_sel}..."):
                try:
                    result = engine.query_gene(
                        gene_sel,
                        ethnicity_group=eth_group,
                        use_ethnicity_prior=use_eth_prior,
                    )
                except ValueError as err:
                    st.error(str(err))
                    return
        else:
            if not observed and not excluded:
                st.warning("Select at least one observed or excluded phenotype, or choose a gene.")
                return
            with st.spinner("Scoring disease modules..."):
                result = engine.query(
                    observed,
                    excluded,
                    ethnicity_group=eth_group,
                    use_ethnicity_prior=use_eth_prior,
                )
        st.session_state["last_result"] = result
    else:
        result = st.session_state.get("last_result")

    st.markdown("<div style='height:.35rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="card-shell" style="padding:.8rem 1rem;background:var(--teal-l);border-color:#bfece4;color:var(--teal);font-weight:700;">Query complete. Results below.</div>',
        unsafe_allow_html=True,
    )

    # Track 2 — Discovery Panel badge (fires whenever ethnicity is set)
    if eth_group:
        disc_mgr = _load_discovery_manager(engine)
        if disc_mgr is not None:
            eng_genes = set(engine.get_gene_options())
            disc_suggestions = disc_mgr.get_suggestions(
                excluded_genes=eng_genes,
                context={"ethnicity_group": eth_group},
            )
            if disc_suggestions:
                st.markdown("<div style='height:.35rem;'></div>", unsafe_allow_html=True)
                _render_discovery_badge(disc_suggestions, eth_group)

    if gene_sel:
        _render_result(result, observed_hpo_ids=engine.gene_observed_hpo_ids(gene_sel), workup_add_to_query=True)
    else:
        _render_result(result, observed_hpo_ids=observed, workup_add_to_query=True)


def _session_mode() -> None:
    _render_page_header(
        "Interactive Session",
        "Answer one phenotype at a time and watch the posterior sharpen as the engine updates after every response.",
    )
    st.markdown(
        """
        <div class="intro-box">
          This redesign preserves the active-learning flow: one question at a time, ranked by information gain, with live posterior updates and confidence tracking.
        </div>
        """,
        unsafe_allow_html=True,
    )

    session_ctx = (gamma_val, eth_group, use_eth_prior)
    if (
        "sess_obj" not in st.session_state
        or st.session_state.get("sess_ctx") != session_ctx
    ):
        st.session_state.sess_obj = engine.new_session(
            ethnicity_group=eth_group,
            use_ethnicity_prior=use_eth_prior,
        )
        st.session_state.sess_ctx = session_ctx
        st.session_state.sess_history = []
        st.session_state.sess_question = None
        st.session_state.sess_result = None

    sess = st.session_state.sess_obj
    history = st.session_state.sess_history

    top_bar_cols = st.columns([3, 3, 1])
    with top_bar_cols[0]:
        manual_fmt = st.selectbox(
            "Manually add a known phenotype",
            options=[""] + HPO_OPTIONS,
            index=0,
            format_func=lambda value: "-- search to add a known phenotype --" if value == "" else value,
            key="sess_manual",
        )
        st.markdown("Hint: " + _hpo_search_hint_md())
    with top_bar_cols[1]:
        yes_col, no_col = st.columns(2)
        with yes_col:
            if st.button("Add as Present", use_container_width=True) and manual_fmt:
                hid = _hpo_id(manual_fmt)
                sess.answer_yes(hid)
                history.append((hid, engine.get_term_name(hid), "yes"))
                st.session_state.sess_question = None
                st.session_state.sess_result = None
                st.rerun()
        with no_col:
            if st.button("Add as Absent", use_container_width=True) and manual_fmt:
                hid = _hpo_id(manual_fmt)
                sess.answer_no(hid)
                history.append((hid, engine.get_term_name(hid), "no"))
                st.session_state.sess_question = None
                st.session_state.sess_result = None
                st.rerun()
    with top_bar_cols[2]:
        if st.button("Reset", use_container_width=True):
            sess.reset()
            st.session_state.sess_history = []
            st.session_state.sess_question = None
            st.session_state.sess_result = None
            st.rerun()

    if st.session_state.sess_question is None:
        with st.spinner("Finding the most informative next question..."):
            next_question = sess.get_next_question()
            st.session_state.sess_question = next_question
            if sess.observed or sess.excluded:
                st.session_state.sess_result = sess.get_current_result()

    question = st.session_state.sess_question
    tier, bg, color = _ig_style(question.information_gain)
    ig_tip = _ig_tooltip(question.information_gain)

    main_cols = st.columns([2, 1])
    with main_cols[0]:
        st.markdown(
            f"""
            <div class="card-shell" style="padding:1.15rem;border-width:2px;border-color:var(--blue);">
              <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;">
                <div>
                  <div class="label">Current Question</div>
                  <div style="font-family:Outfit,sans-serif;font-size:1.45rem;font-weight:800;color:var(--ink);margin-top:.35rem;line-height:1.2;">{_esc(question.term_name)}</div>
                  <div class="mono" style="font-size:13px;color:var(--ink3);margin-top:.3rem;">{_esc(question.hpo_id)}</div>
                </div>
                <span style="font-size:12px;font-weight:700;padding:.35rem .65rem;border-radius:999px;background:{bg};color:{color};white-space:nowrap;">{tier} IG</span>
              </div>
              <div style="margin-top:1rem;padding:.85rem;border-radius:12px;background:#f4f8fc;">
                <div style="font-size:12px;font-weight:600;color:var(--ink2);margin-bottom:.45rem;">Expected information gain</div>
                <div style="display:flex;gap:.65rem;align-items:center;">
                  <div style="flex:1;height:6px;border-radius:999px;background:#dde4ef;overflow:hidden;">
                    <div style="width:{min(question.information_gain / _MAX_ENTROPY_NATS * 100, 100):.2f}%;height:100%;background:{color};"></div>
                  </div>
                  <span class="mono" style="font-size:13px;font-weight:700;color:{color};">{question.information_gain:.3f} nats</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:.3rem;font-size:11px;color:var(--ink3);">
                  <span>0</span>
                  <span>max = ln(17) = 2.833 nats</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(ig_tip)

        answer_cols = st.columns(3)
        with answer_cols[0]:
            if st.button("Yes - Present", type="primary", use_container_width=True):
                sess.answer_yes(question.hpo_id)
                history.append((question.hpo_id, question.term_name, "yes"))
                st.session_state.sess_question = None
                st.session_state.sess_result = None
                st.rerun()
        with answer_cols[1]:
            if st.button("No - Absent", use_container_width=True):
                sess.answer_no(question.hpo_id)
                history.append((question.hpo_id, question.term_name, "no"))
                st.session_state.sess_question = None
                st.session_state.sess_result = None
                st.rerun()
        with answer_cols[2]:
            if st.button("Skip", use_container_width=True):
                if question.hpo_id not in sess.excluded:
                    sess.excluded.append(question.hpo_id)
                history.append((question.hpo_id, question.term_name, "skip"))
                st.session_state.sess_question = None
                st.session_state.sess_result = None
                st.rerun()

        if st.session_state.sess_result is not None:
            st.markdown("<div style='height:.8rem;'></div>", unsafe_allow_html=True)
            _render_label("Current Diagnosis")
            _render_result(
                st.session_state.sess_result,
                show_next_question=False,
                observed_hpo_ids=list(sess.observed),
            )

    with main_cols[1]:
        st.markdown(
            f"""
            <div class="card-shell" style="padding:1rem;">
              <div class="label">Session History ({len(history)})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not history:
            st.markdown(
                '<div style="padding:1rem .25rem;color:var(--ink3);font-size:12px;">No answers recorded yet.</div>',
                unsafe_allow_html=True,
            )
        for index, (hpo_id, name, answer) in enumerate(reversed(history)):
            real_index = len(history) - 1 - index
            if answer == "yes":
                bg_item, fg = "#d1fae5", "#065f46"
            elif answer == "no":
                bg_item, fg = "#ffe4e4", "#9b2c2c"
            else:
                bg_item, fg = "#f0f4f8", "#4a6785"
            row_cols = st.columns([4, 1])
            with row_cols[0]:
                st.markdown(
                    f"""
                    <div style="background:{bg_item};border-radius:10px;padding:.75rem .8rem;">
                      <div style="font-size:12px;font-weight:700;color:{fg};">{_esc(name)}</div>
                      <div class="mono" style="font-size:12px;color:var(--ink3);margin-top:.2rem;">{_esc(hpo_id)} · {answer}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with row_cols[1]:
                if st.button("Undo", key=f"undo_redesign_{hpo_id}_{real_index}", use_container_width=True):
                    if answer == "yes" and hpo_id in sess.observed:
                        sess.observed.remove(hpo_id)
                    elif answer in ("no", "skip") and hpo_id in sess.excluded:
                        sess.excluded.remove(hpo_id)
                    del st.session_state.sess_history[real_index]
                    st.session_state.sess_question = None
                    st.session_state.sess_result = None
                    st.rerun()
        confidence_history = _replay_session_confidences(history)
        if confidence_history:
            _render_confidence_tracker(confidence_history)


def _browser_mode() -> None:
    _render_page_header(
        "Module Browser",
        "Explore phenotype prevalence, discriminating signatures, and gene assignments for each of the 17 IRD modules.",
    )
    st.markdown(
        """
        <div class="intro-box">
          The browser page now uses the redesign tokens and typography while preserving the existing module browsing data source.
        </div>
        """,
        unsafe_allow_html=True,
    )

    selector_col, stat_a, stat_b = st.columns([2.2, 1, 1])
    with selector_col:
        module_id = st.selectbox(
            "Select a disease module",
            options=list(range(17)),
            format_func=lambda module: _module_label(module),
        )
    with st.spinner(f"Loading Module {module_id}..."):
        data = engine.browse_module(module_id)
    with stat_a:
        st.markdown(
            f'<div class="card-shell" style="padding:1rem;text-align:center;"><div class="label">Genes</div><div style="font-family:Outfit,sans-serif;font-size:1.5rem;font-weight:800;color:var(--ink);margin-top:.35rem;">{data["gene_count"]}</div></div>',
            unsafe_allow_html=True,
        )
    with stat_b:
        st.markdown(
            f'<div class="card-shell" style="padding:1rem;text-align:center;"><div class="label">HPO Terms</div><div style="font-family:Outfit,sans-serif;font-size:1.5rem;font-weight:800;color:var(--teal);margin-top:.35rem;">{data["annotated_term_count"]}</div></div>',
            unsafe_allow_html=True,
        )

    tab_terms, tab_sigs, tab_genes, tab_system = st.tabs(
        [
            f"HPO Profile ({data['annotated_term_count']})",
            f"Signatures ({data['signature_count']})",
            f"Gene List ({data['gene_count']})",
            "System Architecture",
        ]
    )

    with tab_terms:
        _render_explain("Phenotypes are sorted by prevalence within the selected module.")
        term_cards = []
        for row in data["top_terms"][:25]:
            prevalence = str(row["Module prevalence"]).replace("%", "")
            try:
                width = float(prevalence)
            except ValueError:
                width = 0.0
            term_cards.append(
                f"""
                <div style="display:flex;align-items:center;gap:.75rem;padding:.8rem .9rem;border-bottom:1px solid #eef2f8;">
                  <span class="mono" style="width:106px;font-size:12px;color:var(--ink3);">{_esc(row['HPO ID'])}</span>
                  <span style="flex:1;font-size:14px;font-weight:500;color:var(--ink);">{_esc(row['Term'])}</span>
                  <div style="display:flex;gap:.5rem;align-items:center;">
                    <div style="width:72px;height:5px;border-radius:999px;background:#dde4ef;overflow:hidden;">
                      <div style="width:{width:.1f}%;height:100%;background:var(--teal);"></div>
                    </div>
                    <span class="mono" style="font-size:12px;font-weight:700;color:var(--teal);width:42px;text-align:right;">{_esc(row['Module prevalence'])}</span>
                  </div>
                </div>
                """
            )
        st.html(f'<div class="card-shell" style="padding:0;overflow:hidden;">{"".join(term_cards)}</div>')

    with tab_sigs:
        _render_explain("Statistically enriched phenotypes vs. all other modules. High odds ratio indicates diagnostic specificity.")
        if data["signatures"]:
            sig_cards = []
            for row in data["signatures"]:
                sig_cards.append(
                    f"""
                    <div style="padding:1rem;border-bottom:1px solid #eef2f8;">
                      <div style="display:flex;align-items:flex-start;gap:1rem;">
                        <div style="flex:1;">
                          <div style="font-size:14px;font-weight:600;color:var(--ink);">{_esc(row['Term'])}</div>
                          <div class="mono" style="font-size:12px;color:var(--ink3);margin-top:.2rem;">{_esc(row['HPO ID'])}</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-family:Outfit,sans-serif;font-size:1.4rem;font-weight:800;color:var(--blue);">{row['Odds Ratio']}</div>
                          <div style="font-size:11px;color:var(--ink3);">Odds ratio</div>
                        </div>
                        <div style="text-align:right;">
                          <div class="mono" style="font-size:13px;font-weight:700;color:var(--emerald);">{_esc(row['q-value'])}</div>
                          <div style="font-size:11px;color:var(--ink3);">FDR</div>
                        </div>
                      </div>
                    </div>
                    """
                )
            st.html(f'<div class="card-shell" style="padding:0;overflow:hidden;">{"".join(sig_cards)}</div>')
        else:
            st.info("No significant module-specific signatures identified.")

    with tab_genes:
        _render_explain("Module-assigned genes retain the existing stability-based ranking from the backend.")
        if data["genes"]:
            gene_cards = []
            for row in data["genes"]:
                stability = _stability_style(str(row["Stability"]))
                gene_cards.append(
                    f"""
                    <div style="display:flex;align-items:center;gap:1rem;padding:.85rem .95rem;border-bottom:1px solid #eef2f8;">
                      <div class="mono" style="width:90px;font-size:13px;font-weight:700;color:var(--ink);">{_esc(row['Gene'])}</div>
                      <div style="flex:1;">
                        <span style="font-size:12px;font-weight:600;padding:.28rem .6rem;border-radius:999px;background:{stability['bg']};color:{stability['color']};">{_esc(stability['label'])}</span>
                      </div>
                      <div class="mono" style="font-size:12px;color:var(--ink2);width:96px;text-align:right;">{row['Stability score']:.4f}</div>
                      <div style="font-size:12px;color:var(--ink3);width:100px;text-align:right;">{row['HPO annotations']} HPO</div>
                    </div>
                    """
                )
            st.html(f'<div class="card-shell" style="padding:0;overflow:hidden;">{"".join(gene_cards)}</div>')
        else:
            st.info("No genes assigned to this module.")

    with tab_system:
        _render_explain(
            "Network-level visualisation of the selected module. Each node represents an IRD gene; "
            "edges reflect shared phenotypic and protein-interaction evidence used during clustering. "
            "Node colour encodes cluster confidence (core / peripheral / unstable); "
            "node size scales with the number of annotated HPO terms."
        )
        system_plot_path = f"Input/single_system_plots/system_plot_module_{module_id}.png"
        if os.path.exists(system_plot_path):
            st.image(
                system_plot_path,
                caption=f"System architecture — Module {module_id}: {_module_label(module_id)}",
                use_container_width=True,
            )
            with open(system_plot_path, "rb") as _f:
                st.download_button(
                    label=f"Download — Module {module_id} system plot",
                    data=_f.read(),
                    file_name=f"system_plot_module_{module_id}.png",
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            st.info(f"No system plot available for Module {module_id}.")

        st.divider()
        _render_label("All-Module Overview")
        _render_explain(
            "Global view of all IRD modules in the network landscape. "
            "Use this to orient the selected module within the broader IRD disease space."
        )
        all_plot_path = "Input/single_system_plots/system_plot_module_All.png"
        if os.path.exists(all_plot_path):
            st.image(
                all_plot_path,
                caption="Full IRD gene network — all modules shown simultaneously. "
                        "Colour encodes module identity; layout reflects phenotypic and network proximity.",
                use_container_width=True,
            )
            with open(all_plot_path, "rb") as _f:
                st.download_button(
                    label="Download — All-module overview",
                    data=_f.read(),
                    file_name="system_plot_module_All.png",
                    mime="image/png",
                    use_container_width=True,
                )


def _analytics_mode():
    _render_page_header(
        "Comparative Analytics",
        "Validation and enrichment analysis of the IRD modules against curated external references.",
    )
    perspective = st.segmented_control(
        "Choose Analysis Perspective",
        options=["Inter-Module Analysis", "Global Context (IRD vs. Universe)"],
        selection_mode="single",
        default="Inter-Module Analysis",
        label_visibility="collapsed",
        key="analytics_perspective_redesign",
    )

    if perspective == "Inter-Module Analysis":
        summary_path = "Input/modules_RetiGene_Comparison/enrichment_results_260412_1551.csv"
        dotplot_path = "Input/modules_RetiGene_Comparison/enrichment_dotplot_260412_1551.png"
        stats = st.columns(4)
        values = [
            ("Validation cases", "9/9", "var(--emerald)", "var(--emerald-l)"),
            ("Avg Recall", "86.2%", "var(--teal)", "var(--teal-l)"),
            ("Avg Precision", "94.1%", "var(--blue)", "var(--blue-l)"),
            ("Novel candidates", "+15", "var(--amber)", "var(--amber-l)"),
        ]
        for col, (label, value, color, bg) in zip(stats, values):
            with col:
                st.markdown(
                    f'<div class="card-shell" style="padding:1rem;text-align:center;background:{bg};"><div style="font-family:Outfit,sans-serif;font-size:1.5rem;font-weight:800;color:{color};">{value}</div><div style="font-size:12px;color:var(--ink2);margin-top:.2rem;">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            rename_map = {
                "module_id": "Module",
                "category_group": "Category_group",
                "value": "Value",
                "overlap": "Overlap",
                "fold_enrichment": "Fold_Enrichment",
                "novel_candidates": "N_novel_candidates",
                "potential_misclassifications": "N_potential_misclassifications",
            }
            df = df.rename(columns=rename_map)
            if "Recall" not in df.columns:
                df["Recall"] = (df["Overlap"] / df["N_category"]) * 100
            if "Precision" not in df.columns:
                df["Precision"] = (df["Overlap"] / df["N_module"]) * 100
            if df["Recall"].dtype == object:
                df["Recall"] = df["Recall"].str.rstrip("%").astype(float)
            if df["Precision"].dtype == object:
                df["Precision"] = df["Precision"].str.rstrip("%").astype(float)
            df["Clinical Module"] = df["Module"].map(lambda module: _module_label(int(module)))
            cols = ["Clinical Module"] + [column for column in df.columns if column not in ["Module", "Clinical Module"]]
            df = df[cols]
            _render_label("RetiGene Atlas Validation")
            _render_explain("Recall and Precision quantify alignment between HPO-based clusters and curated RetiGene annotations.")
            st.dataframe(df, use_container_width=True, hide_index=True, height=480)
        else:
            st.warning("RetiGene enrichment summary file not found.")

        if os.path.exists(dotplot_path):
            st.image(dotplot_path, caption="Visual summary of module-category associations.", use_container_width=True)

        st.divider()
        _render_label("Module Coherence Landscape")
        _render_explain(
            "Within-background similarity of genes to their assigned module, measured in two independent spaces: "
            "phenotypic (ΔHPO, cosine similarity to cluster HPO centroid) and network (ΔNPP, cosine similarity to "
            "cluster protein-interaction centroid). Higher values indicate stronger functional coherence. "
            "Point size in the scatter encodes module gene count; dashed lines mark coherence thresholds."
        )

        dual_scatter_path = "Input/modules_RetiGene_Comparison/dual_coherence_scatter_20260421_1807.png"
        if os.path.exists(dual_scatter_path):
            st.image(
                dual_scatter_path,
                caption="Dual-Coherence Landscape — each module positioned by its median ΔHPO (x) and median ΔNPP (y). "
                        "Modules in the upper-right quadrant are coherent in both phenotypic and network space.",
                use_container_width=True,
            )

        hpo_strip_path = "Input/modules_RetiGene_Comparison/hpo_coherence_stripplot_260423_1021.png"
        npp_strip_path = "Input/modules_RetiGene_Comparison/npp_coherence_stripplot_260421_1625.png"

        _render_label("HPO Coherence (ΔHPO)")
        if os.path.exists(hpo_strip_path):
            st.image(
                hpo_strip_path,
                caption="Per-gene ΔHPO distribution across modules, ranked by median. "
                        "Marker shape encodes cluster stability (core / peripheral / unstable).",
                use_container_width=True,
            )
            with open(hpo_strip_path, "rb") as _f:
                st.download_button(
                    label="Download — HPO Coherence (ΔHPO)",
                    data=_f.read(),
                    file_name="hpo_coherence_stripplot.png",
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            st.warning("HPO coherence stripplot not found.")

        _render_label("NPP Coherence (ΔNPP)")
        if os.path.exists(npp_strip_path):
            st.image(
                npp_strip_path,
                caption="Per-gene ΔNPP distribution across modules, ranked by median. "
                        "Marker fill encodes permutation p-value significance tier.",
                use_container_width=True,
            )
            with open(npp_strip_path, "rb") as _f:
                st.download_button(
                    label="Download — NPP Coherence (ΔNPP)",
                    data=_f.read(),
                    file_name="npp_coherence_stripplot.png",
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            st.warning("NPP coherence stripplot not found.")
    else:
        global_csv_path = "Input/IRD_vs_Non-IRD/IRD_vs_Non-IRD.csv"
        if os.path.exists(global_csv_path):
            df_global = pd.read_csv(global_csv_path)
            column_mapping = {
                "hpo_name": "Phenotype Name",
                "target_count": "IRD Gene Count",
                "bg_count": "Global Gene Count",
                "pct_target": "Prevalence in IRD (%)",
                "pct_bg": "Prevalence Globally (%)",
                "odds_ratio": "Enrichment Magnitude (OR)",
                "q_value": "Significance (FDR)",
                "IC": "Info Content",
                "depth": "HPO Depth",
                "HPO_ID": "HPO ID",
            }
            df_disp = df_global[list(column_mapping.keys())].rename(columns=column_mapping)
            df_disp["Enrichment Magnitude (OR)"] = df_disp["Enrichment Magnitude (OR)"].replace([np.inf, -np.inf], 100.0)
            df_disp["Prevalence in IRD (%)"] = df_disp["Prevalence in IRD (%)"] * 100
            df_disp["Prevalence Globally (%)"] = df_disp["Prevalence Globally (%)"] * 100
            df_disp = df_disp.sort_values("Enrichment Magnitude (OR)", ascending=False)
            threshold = df_disp["Enrichment Magnitude (OR)"].quantile(0.75)
            df_disp = df_disp[df_disp["Enrichment Magnitude (OR)"] >= threshold]
            _render_label("IRD vs. Gene Universe")
            _render_explain("The table is filtered to the top quartile of globally enriched phenotypes.")
            st.dataframe(df_disp, use_container_width=True, hide_index=True, height=560)
        else:
            st.warning("Global enrichment data file not found.")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-sub">
        <b style="color:#5e89ab;">System Specifications</b><br>
        Model: Naive Bayes module inference<br>
        Gene score: SMA-GS with module leakage<br>
        Scope: 17 disease modules, 442 IRD genes<br>
        Optional layer: ethnicity likelihood ratios
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown(
        """
        <div class="sidebar-sub">
        <b style="color:#5e89ab;">Data Provenance</b><br>
        HPO Release: 2026-04-13<br>
        Gene phenotypes: HPO gene annotations<br>
        Module definitions: MPV network clustering<br>
        Ethnicity layer: solved-case LR/count matrices
        </div>
        """,
        unsafe_allow_html=True,
    )

_render_topbar()

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch to active mode
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.app_mode == "Phenotype Query":
    _query_mode()
elif st.session_state.app_mode == "Interactive Session":
    _session_mode()
elif st.session_state.app_mode == "Module Browser":
    _browser_mode()
else:
    _analytics_mode()
