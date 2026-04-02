"""
DeepWatch v2.0 — Enterprise AI Security Platform
=================================================
Full-featured AI surveillance dashboard — Enterprise Edition
UI: Multinational company-standard design (Bosch/Honeywell/Palantir inspired)
    Pure light theme · Zero dark/black colors · Professional blue accent system

Run:
    pip install streamlit streamlit-webrtc ultralytics opencv-python-headless \
                av numpy Pillow scipy
    streamlit run app.py
"""

# ─────────────────────────────── IMPORTS ──────────────────────────────────────
import os, cv2, av, time, base64, smtplib, datetime, json, csv, io, math, sqlite3
import threading
import numpy as np
import streamlit as st
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
from collections import defaultdict, deque
from pathlib import Path

# ─────────────────────────── PAGE CONFIG ──────────────────────────────────────
st.set_page_config(
    page_title="DeepWatch Enterprise — AI Security Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "DeepWatch v2.0 Enterprise — AI-Powered Security Intelligence"},
)

# ─────────────────────────────── CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=IBM+Plex+Mono:wght@400;500;600&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,300&display=swap');

:root {
    --bg-base:       #F5F6FA;
    --bg-surface:    #FFFFFF;
    --bg-raised:     #FAFBFE;
    --bg-subtle:     #EEF0F8;
    --bg-muted:      #E8EAF4;
    --border-light:  rgba(37,99,235,0.08);
    --border-mid:    rgba(37,99,235,0.15);
    --border-strong: rgba(37,99,235,0.25);
    --text-primary:  #0F1729;
    --text-secondary:#3D4E6B;
    --text-tertiary: #7B8DB0;
    --text-muted:    #A8B4CC;
    --brand-50:  #EFF4FF;
    --brand-100: #DBEAFE;
    --brand-400: #60A5FA;
    --brand-500: #3B82F6;
    --brand-600: #2563EB;
    --brand-700: #1D4ED8;
    --success-50:  #F0FDF4;
    --success-500: #22C55E;
    --success-600: #16A34A;
    --warning-50:  #FFFBEB;
    --warning-500: #F59E0B;
    --warning-600: #D97706;
    --danger-50:   #FFF1F2;
    --danger-400:  #FB7185;
    --danger-500:  #EF4444;
    --danger-600:  #DC2626;
    --teal-500:  #14B8A6;
    --violet-500: #8B5CF6;
    --shadow-xs: 0 1px 2px rgba(15,23,41,0.04);
    --shadow-sm: 0 1px 3px rgba(15,23,41,0.06),0 1px 2px rgba(15,23,41,0.04);
    --shadow-md: 0 4px 6px rgba(15,23,41,0.05),0 2px 4px rgba(15,23,41,0.04);
    --shadow-lg: 0 10px 15px rgba(15,23,41,0.06),0 4px 6px rgba(15,23,41,0.04);
    --shadow-xl: 0 20px 25px rgba(15,23,41,0.07),0 8px 10px rgba(15,23,41,0.04);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 18px;
}

*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }

html,body,.stApp {
    background: var(--bg-base) !important;
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 14px; line-height: 1.5;
    -webkit-font-smoothing: antialiased;
}

.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: -3;
    background:
        radial-gradient(ellipse 900px 600px at 0% -100px,rgba(59,130,246,0.05) 0%,transparent 60%),
        radial-gradient(ellipse 600px 400px at 100% 100%,rgba(20,184,166,0.04) 0%,transparent 55%),
        radial-gradient(ellipse 400px 300px at 60% 40%,rgba(245,158,11,0.025) 0%,transparent 60%);
    background-color: var(--bg-base);
}
.stApp::after {
    content: '';
    position: fixed; inset: 0; z-index: -2; pointer-events: none;
    background-image:
        linear-gradient(rgba(59,130,246,0.032) 1px,transparent 1px),
        linear-gradient(90deg,rgba(59,130,246,0.032) 1px,transparent 1px);
    background-size: 56px 56px;
}

.main .block-container { padding:0.5rem 1.25rem 1.5rem; max-width:100%; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-light) !important;
    box-shadow: var(--shadow-lg) !important;
    width: 280px !important;
}
section[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
section[data-testid="stSidebar"] label {
    font-size:.75rem !important; font-weight:500 !important;
    color:var(--text-tertiary) !important; letter-spacing:.2px !important;
}
section[data-testid="stSidebar"] .stSlider { padding-bottom:4px !important; }
section[data-testid="stSidebar"]::before {
    content:'';
    position:absolute; top:0; left:0; width:100%; height:3px;
    background:linear-gradient(90deg,var(--brand-600),var(--teal-500),var(--warning-500));
}

/* ── TOPBAR ── */
.topbar {
    display:flex; align-items:center; justify-content:space-between;
    background:var(--bg-surface);
    border:1px solid var(--border-light);
    border-radius:var(--radius-xl);
    padding:13px 26px; margin-bottom:14px;
    box-shadow:var(--shadow-md);
    position:relative; overflow:hidden;
}
.topbar::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:2.5px;
    background:linear-gradient(90deg,
        var(--brand-600) 0%,var(--brand-400) 35%,
        var(--teal-500) 65%,var(--warning-500) 100%);
}
.topbar-left { display:flex; align-items:center; gap:16px; }
.topbar-logo {
    font-family:'Fraunces',serif;
    font-size:1.35rem; font-weight:700;
    color:var(--text-primary); letter-spacing:-.3px;
    display:flex; align-items:center; gap:10px;
}
.topbar-logo-icon {
    width:34px; height:34px; border-radius:9px;
    background:linear-gradient(135deg,var(--brand-600),var(--brand-400));
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 4px 12px rgba(37,99,235,0.25);
    font-size:.9rem; flex-shrink:0;
}
.topbar-logo em { color:var(--brand-600); font-style:normal; }
.topbar-divider { width:1px; height:24px; background:var(--border-mid); }
.topbar-badge {
    background:var(--brand-50); border:1px solid var(--border-mid);
    border-radius:var(--radius-sm); padding:3px 10px;
    font-family:'IBM Plex Mono',monospace;
    font-size:.62rem; font-weight:500;
    color:var(--brand-600); letter-spacing:.5px;
}
.topbar-badge-warn {
    background:var(--warning-50); border-color:rgba(245,158,11,0.25);
    color:var(--warning-600);
}
.topbar-status {
    display:flex; gap:20px; align-items:center;
    font-family:'IBM Plex Mono',monospace;
    font-size:.67rem; color:var(--text-tertiary);
}
.status-item { display:flex; align-items:center; gap:7px; }
.status-label { color:var(--text-muted); font-size:.62rem; letter-spacing:.5px; }
.status-value { color:var(--text-secondary); font-weight:500; }

.sdot { width:6px; height:6px; border-radius:50%; display:inline-block; flex-shrink:0; }
.sdot-online  { background:var(--success-500); box-shadow:0 0 0 3px rgba(34,197,94,0.15);  animation:pg 2.5s ease infinite; }
.sdot-active  { background:var(--brand-500);   box-shadow:0 0 0 3px rgba(59,130,246,0.15); }
.sdot-warn    { background:var(--warning-500); box-shadow:0 0 0 3px rgba(245,158,11,0.15); animation:py 1.8s ease infinite; }
.sdot-danger  { background:var(--danger-500);  box-shadow:0 0 0 3px rgba(239,68,68,0.15);  animation:pr .8s ease infinite; }
@keyframes pg  { 0%,100%{opacity:1} 50%{opacity:.4} }
@keyframes py  { 0%,100%{opacity:1} 50%{opacity:.3} }
@keyframes pr  { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── METRIC CARDS ── */
.mgrid {
    display:grid; grid-template-columns:repeat(5,1fr);
    gap:12px; margin-bottom:14px;
}
.mcard {
    background:var(--bg-surface);
    border:1px solid var(--border-light);
    border-radius:var(--radius-lg);
    padding:16px 18px 14px;
    position:relative; overflow:hidden;
    box-shadow:var(--shadow-sm);
    transition:box-shadow .2s ease,transform .2s ease,border-color .2s ease;
    cursor:default;
}
.mcard:hover {
    box-shadow:var(--shadow-lg);
    border-color:var(--border-strong);
    transform:translateY(-2px);
}
.mcard::before {
    content:''; position:absolute;
    top:0; left:18px; right:18px; height:2px;
    background:var(--mc,var(--brand-500));
    border-radius:0 0 3px 3px;
}
.mcard::after {
    content:''; position:absolute;
    bottom:-15px; right:-10px;
    width:80px; height:80px; border-radius:50%;
    background:radial-gradient(circle,var(--mc,var(--brand-500)) 0%,transparent 70%);
    opacity:.06;
}
.mcard-icon {
    font-size:.85rem; margin-bottom:8px;
    width:30px; height:30px; border-radius:8px;
    background:var(--mb,var(--brand-50));
    display:flex; align-items:center; justify-content:center;
}
.mval {
    font-family:'Fraunces',serif;
    font-size:2.1rem; font-weight:600;
    color:var(--mc,var(--brand-600));
    line-height:1; margin-bottom:4px; letter-spacing:-.5px;
}
.mlbl {
    font-size:.7rem; font-weight:600;
    color:var(--text-tertiary);
    letter-spacing:.8px; text-transform:uppercase;
}
.msub {
    font-size:.65rem; color:var(--text-muted);
    margin-top:2px; font-family:'IBM Plex Mono',monospace;
}

/* ── PANELS ── */
.panel {
    background:var(--bg-surface);
    border:1px solid var(--border-light);
    border-radius:var(--radius-xl);
    padding:18px 20px 16px; margin-bottom:12px;
    box-shadow:var(--shadow-sm);
    position:relative; overflow:hidden;
}
.panel-hdr {
    display:flex; align-items:center; justify-content:space-between;
    margin-bottom:14px; padding-bottom:12px;
    border-bottom:1px solid var(--border-light);
}
.panel-hdr-left { display:flex; align-items:center; gap:10px; }
.panel-hdr-dot {
    width:8px; height:8px; border-radius:50%;
    background:var(--brand-500);
    box-shadow:0 0 0 3px rgba(59,130,246,0.12);
}
.panel-hdr-title {
    font-family:'Plus Jakarta Sans',sans-serif;
    font-size:.72rem; font-weight:700;
    color:var(--text-secondary);
    letter-spacing:1.5px; text-transform:uppercase;
}
.panel-badge {
    font-family:'IBM Plex Mono',monospace;
    font-size:.6rem; font-weight:500;
    color:var(--brand-600); background:var(--brand-50);
    border:1px solid var(--border-mid);
    border-radius:20px; padding:2px 10px; letter-spacing:.3px;
}
.panel-badge-live {
    color:var(--success-600); background:var(--success-50);
    border-color:rgba(34,197,94,0.2);
    animation:lp 2s ease infinite;
}
@keyframes lp { 0%,100%{opacity:1} 50%{opacity:.55} }

/* ── ALERTS ── */
.alert-critical {
    display:flex; align-items:flex-start; gap:12px;
    background:var(--danger-50);
    border:1px solid rgba(239,68,68,0.22);
    border-left:3px solid var(--danger-500);
    border-radius:var(--radius-md);
    padding:11px 16px; margin:6px 0;
    animation:ap .9s ease infinite alternate;
    box-shadow:0 2px 8px rgba(239,68,68,0.07);
}
.alert-critical-icon { font-size:1rem; flex-shrink:0; margin-top:1px; }
.alert-critical-title {
    font-size:.78rem; font-weight:700;
    color:var(--danger-600);
    font-family:'Plus Jakarta Sans',sans-serif; margin-bottom:2px;
}
.alert-critical-sub {
    font-size:.67rem; color:rgba(220,38,38,0.65);
    font-family:'IBM Plex Mono',monospace;
}
@keyframes ap {
    from { border-left-color:var(--danger-500); box-shadow:0 2px 8px rgba(239,68,68,0.07); }
    to   { border-left-color:var(--danger-400); box-shadow:0 4px 16px rgba(239,68,68,0.16); }
}
.alert-warning {
    display:flex; align-items:center; gap:10px;
    background:var(--warning-50);
    border:1px solid rgba(245,158,11,0.2);
    border-left:3px solid var(--warning-500);
    border-radius:var(--radius-md);
    padding:10px 16px; margin:5px 0;
}
.alert-warning span { font-size:.76rem; font-weight:500; color:var(--warning-600); }

/* ── OBJECT TAGS ── */
.otag {
    display:inline-flex; align-items:center; gap:5px;
    padding:4px 12px; border-radius:20px;
    font-family:'IBM Plex Mono',monospace;
    font-size:.67rem; font-weight:500;
    margin:3px; border:1px solid;
}
.t-person  { color:#2563EB; background:#EFF6FF; border-color:rgba(37,99,235,0.2); }
.t-vehicle { color:#0891B2; background:#ECFEFF; border-color:rgba(8,145,178,0.2); }
.t-animal  { color:#059669; background:#ECFDF5; border-color:rgba(5,150,105,0.2); }
.t-object  { color:#D97706; background:#FFFBEB; border-color:rgba(217,119,6,0.2); }
.t-weapon  { color:#DC2626; background:#FFF1F2; border-color:rgba(220,38,38,0.25); font-weight:600; }

/* ── BARS ── */
.cbar-wrap {
    height:4px; border-radius:4px;
    background:var(--bg-muted); margin-top:5px; overflow:hidden;
}
.cbar { height:4px; border-radius:4px; transition:width .5s cubic-bezier(.4,0,.2,1); }

/* ── DET ITEMS ── */
.det-item {
    padding:8px 10px; border-radius:var(--radius-sm);
    background:var(--bg-raised);
    border:1px solid var(--border-light);
    margin:5px 0; transition:background .15s;
}
.det-item:hover { background:var(--brand-50); border-color:var(--border-mid); }
.det-row {
    display:flex; align-items:center; justify-content:space-between;
    font-family:'IBM Plex Mono',monospace; font-size:.68rem;
}
.det-label { color:var(--text-secondary); font-weight:500; }
.det-conf  { font-weight:600; }

/* ── EVENT LOG ── */
.log-row {
    display:flex; gap:10px; align-items:center;
    padding:8px 12px; border-radius:var(--radius-sm);
    border-left:3px solid; margin:3px 0;
    font-family:'IBM Plex Mono',monospace; font-size:.67rem;
    background:var(--bg-raised);
    border-top:1px solid var(--border-light);
    border-right:1px solid var(--border-light);
    border-bottom:1px solid var(--border-light);
    transition:background .15s;
}
.log-row:hover { background:var(--brand-50); }
.log-t { color:var(--text-muted); min-width:60px; }
.log-m { flex:1; color:var(--text-secondary); }
.log-n { color:var(--brand-600); min-width:28px; text-align:right; font-weight:600; }
.lc { border-left-color:var(--danger-500); }
.lw { border-left-color:var(--warning-500); }
.li { border-left-color:var(--brand-500); }
.ls { border-left-color:var(--success-500); }

/* ── ZONE BADGE ── */
.zone-badge {
    display:flex; align-items:center; gap:10px;
    background:var(--warning-50);
    border:1px solid rgba(245,158,11,0.2);
    border-radius:var(--radius-sm);
    padding:8px 14px; margin:5px 0;
}
.zone-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.zone-text {
    font-family:'IBM Plex Mono',monospace;
    font-size:.7rem; color:var(--warning-600); font-weight:500;
}

/* ── STAT ITEM ── */
.stat-item {
    padding:6px 0;
    border-bottom:1px solid var(--border-light);
}
.stat-item:last-child { border-bottom:none; }
.stat-row {
    display:flex; align-items:center; justify-content:space-between;
    margin-bottom:4px;
}
.stat-label {
    font-size:.71rem; font-weight:500;
    color:var(--text-secondary);
    display:flex; align-items:center; gap:6px;
}
.stat-count {
    font-family:'IBM Plex Mono',monospace;
    font-size:.71rem; font-weight:600;
}

/* ── FPS STRIP ── */
.fps-strip {
    display:flex; gap:6px; align-items:center;
    padding:6px 10px; margin-top:10px;
    background:var(--bg-subtle);
    border:1px solid var(--border-light);
    border-radius:var(--radius-sm);
    font-family:'IBM Plex Mono',monospace;
    font-size:.64rem; color:var(--text-tertiary);
}
.fps-chip {
    background:var(--bg-muted); border-radius:4px;
    padding:2px 8px; color:var(--brand-600); font-weight:600;
}

/* ── BUTTONS ── */
.stButton button {
    background:var(--brand-600) !important;
    color:#fff !important; border:none !important;
    border-radius:var(--radius-md) !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:.76rem !important; font-weight:600 !important;
    padding:8px 18px !important;
    box-shadow:0 2px 8px rgba(37,99,235,0.28) !important;
    transition:all .2s ease !important;
}
.stButton button:hover {
    background:var(--brand-700) !important;
    box-shadow:0 6px 16px rgba(37,99,235,0.35) !important;
    transform:translateY(-1px) !important;
}
.stButton button:active { transform:translateY(0) !important; }

/* ── DOWNLOAD BUTTONS ── */
.stDownloadButton button {
    background:var(--bg-surface) !important;
    color:var(--brand-600) !important;
    border:1.5px solid var(--border-strong) !important;
    border-radius:var(--radius-md) !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:.74rem !important; font-weight:600 !important;
    box-shadow:var(--shadow-xs) !important;
    transition:all .2s ease !important;
}
.stDownloadButton button:hover {
    background:var(--brand-50) !important;
    border-color:var(--brand-400) !important;
    box-shadow:var(--shadow-md) !important;
    transform:translateY(-1px) !important;
}

/* ── SIDEBAR SECTION HEADER ── */
.sb-section {
    font-family:'Plus Jakarta Sans',sans-serif;
    font-size:.63rem; font-weight:700;
    color:var(--text-tertiary) !important;
    letter-spacing:1.5px; text-transform:uppercase;
    padding:4px 0 2px;
    display:flex; align-items:center; gap:8px;
}
.sb-section::after { content:''; flex:1; height:1px; background:var(--border-light); }
section[data-testid="stSidebar"] strong {
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:700 !important; font-size:.68rem !important;
    color:var(--text-tertiary) !important;
    letter-spacing:1.2px !important; text-transform:uppercase !important;
}

/* ── FORM ELEMENTS ── */
.stCheckbox label { font-size:.75rem !important; font-weight:500 !important; }
div[data-baseweb="select"]>div {
    border-color:var(--border-mid) !important;
    border-radius:var(--radius-sm) !important;
    background:var(--bg-surface) !important;
    box-shadow:var(--shadow-xs) !important;
}
.stNumberInput input {
    border-color:var(--border-mid) !important;
    border-radius:var(--radius-sm) !important;
    font-family:'IBM Plex Mono',monospace !important;
}
div[data-baseweb="slider"] { margin-top:-4px !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background:var(--bg-subtle) !important;
    border-radius:var(--radius-md) !important;
    padding:3px !important; gap:2px !important;
    border:1px solid var(--border-light) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius:var(--radius-sm) !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:.73rem !important; font-weight:500 !important;
    color:var(--text-tertiary) !important;
    padding:5px 16px !important;
}
.stTabs [aria-selected="true"] {
    background:var(--bg-surface) !important;
    color:var(--brand-600) !important;
    box-shadow:var(--shadow-sm) !important;
    font-weight:600 !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:var(--bg-base); }
::-webkit-scrollbar-thumb { background:var(--border-strong); border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:var(--brand-400); }

hr { border:none !important; border-top:1px solid var(--border-light) !important; margin:10px 0 !important; }

.model-info-box {
    background:var(--brand-50); border:1px solid var(--border-mid);
    border-radius:var(--radius-sm); padding:8px 12px;
    font-family:'IBM Plex Mono',monospace;
    font-size:.62rem; color:var(--text-tertiary); line-height:1.7; margin:4px 0 6px;
}
.model-info-box span { color:var(--brand-600); font-weight:600; }
.model-warn-box {
    background:var(--warning-50); border:1px solid rgba(245,158,11,0.25);
    border-radius:var(--radius-sm); padding:8px 12px; margin-top:5px;
    font-family:'IBM Plex Mono',monospace;
    font-size:.62rem; color:var(--warning-600); line-height:1.6;
}
.email-info-box {
    background:var(--brand-50); border:1px solid var(--border-mid);
    border-radius:var(--radius-sm); padding:9px 12px;
    font-family:'IBM Plex Mono',monospace;
    font-size:.62rem; color:var(--text-tertiary); line-height:1.8;
}
.email-info-box .key { color:var(--brand-600); font-weight:600; }

.vfooter {
    text-align:center; padding:.9rem 1rem; margin-top:12px;
    border-top:1px solid var(--border-light);
    font-family:'IBM Plex Mono',monospace;
    font-size:.6rem; letter-spacing:1.5px; color:var(--text-muted);
}
.vfooter strong { color:var(--text-tertiary); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────── DATABASE (SQLite) ────────────────────────────────────
DB_PATH = Path("deepwatch_events.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, level TEXT, category TEXT, label TEXT,
            message TEXT, conf REAL, cam TEXT
        )
    """)
    con.commit(); con.close()

def db_insert(level, category, label, message, conf=0.0, cam="CAM-01"):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO events (ts,level,category,label,message,conf,cam) VALUES (?,?,?,?,?,?,?)",
            (datetime.datetime.now().isoformat(), level, category, label, message, conf, cam)
        )
        con.commit(); con.close()
    except Exception: pass

def db_fetch_recent(n=100):
    try:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT ts,level,category,label,message,conf,cam FROM events ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        con.close(); return rows
    except Exception: return []

def db_export_csv():
    rows = db_fetch_recent(5000)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp","level","category","label","message","conf","cam"])
    w.writerows(rows)
    return buf.getvalue().encode()

def db_export_json():
    rows = db_fetch_recent(5000)
    data = [{"ts":r[0],"level":r[1],"category":r[2],"label":r[3],
              "message":r[4],"conf":r[5],"cam":r[6]} for r in rows]
    return json.dumps(data, indent=2).encode()

init_db()


# ─────────────────────────── WEBRTC CONFIG ────────────────────────────────────
def _env(k, d=""):
    try: return st.secrets.get(k, os.getenv(k, d))
    except Exception: return os.getenv(k, d)

_turn_urls = [u for u in [_env("TURN_URL_1"),_env("TURN_URL_2"),
                            _env("TURN_URL_3"),_env("TURN_URL_4")] if u.strip()]
_tu = _env("TURN_USERNAME"); _tp = _env("TURN_PASSWORD")

if _turn_urls and _tu and _tp:
    _ice = [{"urls":["stun:stun.relay.metered.ca:80"]},
            {"urls":_turn_urls,"username":_tu,"credential":_tp}]
else:
    _ice = [{"urls":["stun:stun.l.google.com:19302"]},
            {"urls":["stun:stun1.l.google.com:19302"]},
            {"urls":["turn:openrelay.metered.ca:80",
                     "turn:openrelay.metered.ca:443",
                     "turn:openrelay.metered.ca:443?transport=tcp"],
             "username":"openrelayproject","credential":"openrelayproject"}]

_rtc_cfg_dict = {"iceServers": _ice}
if _env("FORCE_TURN","false").lower()=="true":
    _rtc_cfg_dict["iceTransportPolicy"] = "relay"
RTC_CONFIG = RTCConfiguration(_rtc_cfg_dict)


# ─────────────────────────── YOLO MODEL ──────────────────────────────────────
@st.cache_resource
def load_model(weights: str = "yolov8n.pt"):
    if not Path(weights).exists() and weights not in ["yolov8n.pt","yolov8s.pt","yolov8m.pt"]:
        return None
    return YOLO(weights)

MODEL_OPTIONS = {
    "YOLOv8n · General (fastest)":  {"file":"yolov8n.pt","is_weapon_model":False,"desc":"80 COCO classes · people, cars, animals…"},
    "YOLOv8s · General (balanced)": {"file":"yolov8s.pt","is_weapon_model":False,"desc":"80 COCO classes · better accuracy"},
    "YOLOv8m · General (accurate)": {"file":"yolov8m.pt","is_weapon_model":False,"desc":"80 COCO classes · highest accuracy"},
    "🔫 WeaponV1 · 14 Classes":     {"file":"weapon_v1.pt","is_weapon_model":True,"desc":"AK47 · Rifle · Revolver · Shotgun · Knife · Axe · Sword · M16…"},
    "🔫 WeaponV2 · 5 Classes":      {"file":"weapon_v2.pt","is_weapon_model":True,"desc":"Pistol · Rifle · Knife · Grenade · Missile"},
    "🔫 WeaponV3 · Gun+Knife":      {"file":"weapon_v3.pt","is_weapon_model":True,"desc":"Handgun · Shotgun · Knife · Rifle"},
}
WEAPON_KEYWORDS = {"gun","pistol","rifle","revolver","shotgun","ak47","m16","firearm",
    "weapon","knife","sword","axe","grenade","missile","handgun","sniper",
    "carbine","uzi","glock","assault","explosive","bomb","blade"}
PERSON_IDS  = {0}
VEHICLE_IDS = {1,2,3,5,7}
ANIMAL_IDS  = set(range(14,24))
WEAPON_IDS  = {76,43}
CAT_BGR = {"person":(80,80,255),"vehicle":(255,200,0),"animal":(80,255,128),"object":(0,200,255),"weapon":(0,0,255)}
CAT_HEX = {"person":"#2563EB","vehicle":"#0891B2","animal":"#059669","object":"#D97706","weapon":"#DC2626"}
CAT_ICON= {"person":"👤","vehicle":"🚗","animal":"🐾","object":"📦","weapon":"⚠️"}

def get_category(cid,label="",is_weapon_model=False):
    if is_weapon_model: return "weapon"
    if label.lower() in WEAPON_KEYWORDS: return "weapon"
    if cid in PERSON_IDS:  return "person"
    if cid in VEHICLE_IDS: return "vehicle"
    if cid in ANIMAL_IDS:  return "animal"
    if cid in WEAPON_IDS:  return "weapon"
    return "object"


# ─────────────────────────── EMAIL HELPER ────────────────────────────────────
def send_email_alert(subject, body, img_bytes=None):
    smtp_host = _env("SMTP_HOST","smtp.gmail.com")
    smtp_port = int(_env("SMTP_PORT","587"))
    smtp_user = _env("SMTP_USER","")
    smtp_pass = _env("SMTP_PASS","")
    smtp_to   = _env("SMTP_TO", smtp_user)
    if not smtp_user or not smtp_pass:
        return False,"SMTP credentials not configured"
    try:
        msg = MIMEMultipart()
        msg["From"]    = smtp_user
        msg["To"]      = smtp_to
        msg["Subject"] = f"[DeepWatch 🚨] {subject}"
        msg.attach(MIMEText(
            f"{body}\n\n────────────────────────────\n"
            f"DeepWatch v2.0 Enterprise | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"This is an automated security alert.\n","plain"))
        if img_bytes:
            att = MIMEImage(img_bytes, name="deepwatch_alert.jpg")
            att.add_header("Content-Disposition","attachment",filename="deepwatch_alert.jpg")
            msg.attach(att)
        with smtplib.SMTP(smtp_host,smtp_port,timeout=10) as srv:
            srv.starttls(); srv.login(smtp_user,smtp_pass); srv.send_message(msg)
        return True,"OK"
    except Exception as exc:
        return False,str(exc)


# ─────────────────────── SESSION STATE INIT ──────────────────────────────────
_defaults = {
    "total_detections":0,"people_count":0,"vehicle_count":0,
    "alert_count":0,"snap_count":0,"email_count":0,
    "label_counts":defaultdict(int),"event_log":deque(maxlen=300),
    "snapshots":deque(maxlen=12),"heatmap":None,"loiter_times":{},
    "start_time":time.time(),"last_alert_time":0.0,
    "last_snap_time":0.0,"last_weapon_alert":0.0,
    "alert_cats":{"person"},"snap_cats":{"person"},
    "boundary_zones":[],"loiter_threshold":8,
    "model_key":"YOLOv8n · General (fastest)",
}
for k,v in _defaults.items():
    if k not in st.session_state: st.session_state[k] = v


# ──────────────────────────── SIDEBAR ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:14px 4px 10px;display:flex;align-items:center;gap:10px">
      <div style="width:28px;height:28px;border-radius:7px;
                  background:linear-gradient(135deg,#2563EB,#60A5FA);
                  display:flex;align-items:center;justify-content:center;
                  font-size:.75rem;box-shadow:0 3px 8px rgba(37,99,235,0.25)">🛡️</div>
      <div>
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:.82rem;
                    font-weight:700;color:#0F1729;letter-spacing:.3px">Control Panel</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:.57rem;
                    color:#A8B4CC;letter-spacing:1.2px">DEEPWATCH ENTERPRISE</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sb-section">📷 Camera</div>', unsafe_allow_html=True)
    camera_facing = st.selectbox("Feed",["Back (Rear)","Front (Selfie)"],index=0,label_visibility="collapsed")
    model_key = st.selectbox("Detection Model",list(MODEL_OPTIONS.keys()),index=0)
    st.session_state.model_key = model_key

    _minfo = MODEL_OPTIONS[model_key]; _is_weapon = _minfo["is_weapon_model"]
    st.markdown(
        f'<div class="model-info-box"><span>{"⚠ WEAPON MODEL" if _is_weapon else "◉ GENERAL MODEL"}</span><br>{_minfo["desc"]}</div>',
        unsafe_allow_html=True)

    _model_file = _minfo["file"]
    _model_missing = (_is_weapon and not Path(_model_file).exists())
    if _model_missing:
        st.markdown(f'<div class="model-warn-box">⚠ File not found: <b>{_model_file}</b><br>Falling back to YOLOv8n</div>',unsafe_allow_html=True)
        model_obj = load_model("yolov8n.pt")
    else:
        model_obj = load_model(_model_file)
    IS_WEAPON_MODEL = _is_weapon and not _model_missing

    conf_thresh = st.slider("Confidence Threshold",0.20,0.90,0.45,0.05,format="%.2f")
    max_det     = st.slider("Max Detections / Frame",5,50,20,5)
    st.markdown("---")

    st.markdown('<div class="sb-section">🕐 Operation Mode</div>', unsafe_allow_html=True)
    op_mode = st.radio("Mode",["Always On","Vacation (24/7)","Scheduled"],label_visibility="collapsed")
    active_now = True
    if op_mode == "Scheduled":
        c1,c2 = st.columns(2)
        with c1: t_start = st.time_input("Start",datetime.time(22,0))
        with c2: t_end   = st.time_input("End",datetime.time(6,0))
        now_t = datetime.datetime.now().time()
        active_now = (t_start<=now_t<=t_end) if t_start<t_end else (now_t>=t_start or now_t<=t_end)
    st.markdown("---")

    st.markdown('<div class="sb-section">🚧 Boundary Zones</div>', unsafe_allow_html=True)
    bz_count = st.number_input("Number of zones",0,4,1,1)
    boundary_zones = []
    zone_colors = ["#F59E0B","#EF4444","#059669","#2563EB"]
    zone_names  = ["ZONE-A","ZONE-B","ZONE-C","ZONE-D"]
    for i in range(int(bz_count)):
        y_pct = st.slider(f"{zone_names[i]} height (%)",0,100,65-i*15,5,key=f"zone_{i}")
        boundary_zones.append((y_pct,zone_names[i],zone_colors[i]))
    st.session_state.boundary_zones = boundary_zones
    st.markdown("---")

    st.markdown('<div class="sb-section">🚨 Alert Triggers</div>', unsafe_allow_html=True)
    ac_person  = st.checkbox("Person",         value=True,  key="ac_p")
    ac_vehicle = st.checkbox("Vehicle",        value=False, key="ac_v")
    ac_animal  = st.checkbox("Animal",         value=False, key="ac_a")
    ac_object  = st.checkbox("Object",         value=False, key="ac_o")
    ac_weapon  = st.checkbox("Weapon ⚠",       value=True,  key="ac_w")
    ac_loiter  = st.checkbox("Loitering",      value=True,  key="ac_l")
    ac_crowd   = st.checkbox("Crowd (>N)",     value=False, key="ac_cr")
    crowd_n    = st.slider("Crowd threshold",2,20,5,1) if ac_crowd else 5
    alert_cats = set()
    if ac_person:  alert_cats.add("person")
    if ac_vehicle: alert_cats.add("vehicle")
    if ac_animal:  alert_cats.add("animal")
    if ac_object:  alert_cats.add("object")
    if ac_weapon:  alert_cats.add("weapon")
    st.session_state.alert_cats = alert_cats
    alert_cooldown = st.slider("Alert cooldown (s)",5,120,30,5)
    st.markdown("---")

    st.markdown('<div class="sb-section">⏳ Loitering</div>', unsafe_allow_html=True)
    loiter_thresh = st.slider("Loiter threshold (s)",3,60,8,1)
    st.session_state.loiter_threshold = loiter_thresh
    st.markdown("---")

    st.markdown('<div class="sb-section">📸 Auto Snapshot</div>', unsafe_allow_html=True)
    sc_person  = st.checkbox("On person",  value=True,  key="sc_p")
    sc_vehicle = st.checkbox("On vehicle", value=False, key="sc_v")
    sc_animal  = st.checkbox("On animal",  value=False, key="sc_a")
    sc_weapon  = st.checkbox("On weapon",  value=True,  key="sc_w")
    snap_cats = set()
    if sc_person:  snap_cats.add("person")
    if sc_vehicle: snap_cats.add("vehicle")
    if sc_animal:  snap_cats.add("animal")
    if sc_weapon:  snap_cats.add("weapon")
    st.session_state.snap_cats = snap_cats
    snap_cooldown = st.slider("Snap cooldown (s)",2,60,10,2)
    st.markdown("---")

    st.markdown('<div class="sb-section">📧 Email Alerts</div>', unsafe_allow_html=True)
    email_enabled = st.checkbox("Enable email alerts",value=False)
    if email_enabled:
        st.markdown('<div class="email-info-box">Configure in Streamlit Secrets:<br>'
            '<span class="key">SMTP_USER · SMTP_PASS · SMTP_TO</span><br>'
            'Optional: SMTP_HOST · SMTP_PORT<br>Gmail: use an App Password</div>',
            unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sb-section">🎨 Overlay</div>', unsafe_allow_html=True)
    show_heatmap   = st.checkbox("Show Heatmap",         value=False)
    show_tracks    = st.checkbox("Show Tracking Trails", value=True)
    show_crowd_map = st.checkbox("Show Crowd Density",   value=False)
    show_fps_hud   = st.checkbox("Show HUD Overlay",     value=True)
    draw_style     = st.radio("Box Style",["Corners","Full Box","Dot"],label_visibility="collapsed",horizontal=True)
    st.markdown("---")

    st.markdown('<div class="sb-section">💾 Export</div>', unsafe_allow_html=True)
    ca,cb = st.columns(2)
    with ca: st.download_button("📥 CSV", data=db_export_csv(),file_name="deepwatch_log.csv",mime="text/csv",use_container_width=True)
    with cb: st.download_button("📥 JSON",data=db_export_json(),file_name="deepwatch_log.json",mime="application/json",use_container_width=True)
    st.markdown("---")
    if st.button("↺ Clear Session Data",use_container_width=True):
        for k,v in _defaults.items():
            st.session_state[k]=(v() if callable(v) else v.copy() if isinstance(v,(dict,deque)) else v)
        st.rerun()


IS_WEAPON_MODEL = False

# ─────────────────────────── VIDEO PROCESSOR ──────────────────────────────────
class AdvancedVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._lock=threading.Lock(); self.detections=[]; self.fps=0
        self.frame_count=0; self.last_frame=None; self._t=time.time()
        self._trackers={}; self._next_id=0; self._heatmap=None
        self._heatmap_decay=0.995
        self._trails=defaultdict(lambda: deque(maxlen=30))
        self._h=self._w=0

    def _update_tracker(self,new_centers):
        MATCH_DIST=80; assigned={}; free_ids=list(self._trackers.keys())
        for cx,cy,cat in new_centers:
            best_id,best_d=None,MATCH_DIST+1
            for tid in free_ids:
                tx,ty=self._trackers[tid]["center"]
                d=math.hypot(cx-tx,cy-ty)
                if d<best_d: best_d,best_id=d,tid
            if best_id is not None:
                self._trackers[best_id].update({"center":(cx,cy),"age":0})
                assigned[best_id]=(cx,cy,cat,self._trackers[best_id]["entry"])
                free_ids.remove(best_id)
            else:
                tid=self._next_id; self._next_id+=1
                self._trackers[tid]={"center":(cx,cy),"age":0,"entry":time.time(),"cat":cat}
                assigned[tid]=(cx,cy,cat,self._trackers[tid]["entry"])
        for tid in list(self._trackers.keys()):
            if tid not in assigned:
                self._trackers[tid]["age"]+=1
                if self._trackers[tid]["age"]>15:
                    del self._trackers[tid]
                    if tid in self._trails: del self._trails[tid]
        return assigned

    def _draw_box(self,img,x1,y1,x2,y2,color,label,style):
        if style=="Corners":
            blen=min(14,(x2-x1)//3,(y2-y1)//3)
            for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(img,(px,py),(px+dx*blen,py),color,2)
                cv2.line(img,(px,py),(px,py+dy*blen),color,2)
        elif style=="Full Box":
            cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
        else:
            cx,cy=(x1+x2)//2,(y1+y2)//2
            cv2.circle(img,(cx,cy),6,color,-1); cv2.circle(img,(cx,cy),8,color,1)
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.4,1)
        cv2.rectangle(img,(x1,y1-th-8),(x1+tw+8,y1),color,-1)
        cv2.putText(img,label,(x1+4,y1-4),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)

    def recv(self,frame:av.VideoFrame)->av.VideoFrame:
        img=frame.to_ndarray(format="bgr24")
        self.frame_count+=1
        now=time.time(); self.fps=round(1.0/max(now-self._t,1e-6)); self._t=now
        h,w=img.shape[:2]
        if w>640:
            scale=640/w; img=cv2.resize(img,(640,int(h*scale)))
        self._h,self._w=img.shape[:2]
        if self._heatmap is None or self._heatmap.shape[:2]!=img.shape[:2]:
            self._heatmap=np.zeros((self._h,self._w),dtype=np.float32)

        if self.frame_count%2==0:
            results=model_obj(img,conf=conf_thresh,max_det=max_det,verbose=False)[0]
            new_dets=[]
            for box in results.boxes:
                cid=int(box.cls[0]); conf=float(box.conf[0])
                lbl=model_obj.names[cid]; cat=get_category(cid,lbl,IS_WEAPON_MODEL)
                x1,y1,x2,y2=map(int,box.xyxy[0])
                new_dets.append({"label":lbl,"conf":conf,"category":cat,
                                  "box":(x1,y1,x2,y2),"cx":(x1+x2)//2,"cy":(y1+y2)//2})
            with self._lock: self.detections=new_dets

        with self._lock: dets=list(self.detections)

        self._heatmap*=self._heatmap_decay
        for d in dets:
            if d["category"]=="person":
                cx,cy=d["cx"],d["cy"]; r=30
                self._heatmap[max(0,cy-r):min(self._h,cy+r),max(0,cx-r):min(self._w,cx+r)]+=0.15

        person_centers=[(d["cx"],d["cy"],d["category"]) for d in dets if d["category"]=="person"]
        track_map=self._update_tracker(person_centers)
        for tid,(cx,cy,_,_) in track_map.items():
            self._trails[tid].appendleft((cx,cy))

        if show_heatmap:
            hm_u8=(np.clip(self._heatmap,0,1)*255).astype(np.uint8)
            hm_col=cv2.applyColorMap(hm_u8,cv2.COLORMAP_JET)
            mask=hm_u8>10; img[mask]=cv2.addWeighted(img,0.5,hm_col,0.5,0)[mask]

        zone_breach_labels=[]
        for (y_pct,z_label,z_hex) in st.session_state.get("boundary_zones",[]):
            ly=int(self._h*(y_pct/100))
            z_bgr=tuple(int(z_hex.lstrip("#")[i:i+2],16) for i in (4,2,0))
            x=0
            while x<self._w:
                cv2.line(img,(x,ly),(min(x+20,self._w),ly),z_bgr,2); x+=30
            cv2.putText(img,f"▶ {z_label}",(8,ly-7),cv2.FONT_HERSHEY_SIMPLEX,.42,z_bgr,1,cv2.LINE_AA)
            for d in dets:
                if d["category"]=="person" and d["cy"]>ly:
                    zone_breach_labels.append(f"{z_label} BREACH")

        if show_tracks:
            for tid,trail in self._trails.items():
                pts=list(trail)
                for i in range(1,len(pts)):
                    alpha=1.0-i/len(pts)
                    cv2.line(img,pts[i-1],pts[i],(int(60*alpha),int(130*alpha),int(246*alpha)),1,cv2.LINE_AA)

        loiter_alerts=[]; now_t=time.time()
        for d in dets:
            x1,y1,x2,y2=d["box"]
            col=CAT_BGR.get(d["category"],(180,180,180))
            self._draw_box(img,x1,y1,x2,y2,col,f"{d['label'].upper()} {int(d['conf']*100)}%",draw_style)

        for tid,(cx,cy,cat,entry_t) in track_map.items():
            dur=now_t-entry_t
            if cat=="person" and dur>2:
                col=(0,80,220) if dur>loiter_thresh else (0,160,220)
                cv2.putText(img,f"ID{tid} {int(dur)}s",(cx-20,cy+25),cv2.FONT_HERSHEY_SIMPLEX,.38,col,1,cv2.LINE_AA)
                if dur>loiter_thresh and ac_loiter: loiter_alerts.append(f"LOITERING ID{tid} ({int(dur)}s)")

        n_people=sum(1 for d in dets if d["category"]=="person")
        if show_crowd_map and n_people>0:
            bw=int(min(1.0,n_people/crowd_n)*(self._w-20))
            bc=(80,180,80) if n_people<crowd_n//2 else (60,140,220) if n_people<crowd_n else (60,60,220)
            cv2.rectangle(img,(10,10),(10+bw,20),bc,-1)
            cv2.rectangle(img,(10,10),(self._w-10,20),(180,180,200),1)
            cv2.putText(img,f"CROWD: {n_people}/{crowd_n}",(14,18),cv2.FONT_HERSHEY_SIMPLEX,.38,(80,80,120),1)

        if show_fps_hud:
            ov=img.copy()
            cv2.rectangle(ov,(0,self._h-34),(self._w,self._h),(238,242,252),-1)
            cv2.addWeighted(ov,0.8,img,0.2,0,img)
            cv2.putText(img,
                f"CAM-01  |  {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}  |"
                f"  FPS:{self.fps}  OBJ:{len(dets)}  |  {'ACTIVE' if active_now else 'SLEEP'}",
                (8,self._h-11),cv2.FONT_HERSHEY_SIMPLEX,.35,(60,90,160),1,cv2.LINE_AA)

        rec_col=(60,80,220) if int(time.time()*2)%2==0 else (160,185,235)
        cv2.circle(img,(self._w-16,16),6,rec_col,-1)
        cv2.putText(img,"REC",(self._w-38,20),cv2.FONT_HERSHEY_SIMPLEX,.32,(60,80,200),1,cv2.LINE_AA)

        with self._lock:
            self.last_frame=img.copy()
            self._zone_breaches=zone_breach_labels
            self._loiter_alerts=loiter_alerts
            self._n_people=n_people

        if self.frame_count%30==0:
            hm_u8=(np.clip(self._heatmap,0,1)*255).astype(np.uint8)
            st.session_state["heatmap"]=cv2.applyColorMap(hm_u8,cv2.COLORMAP_JET)

        return av.VideoFrame.from_ndarray(img,format="bgr24")


# ─────────────────────── TOP BAR ─────────────────────────────────────────────
_now_str  = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
_mode_dot = "sdot-active" if active_now else "sdot-warn"
_mode_lbl = "ACTIVE" if active_now else "SLEEP"

st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">
      <div class="topbar-logo-icon">🛡️</div>
      Deep<em>Watch</em>
    </div>
    <div class="topbar-divider"></div>
    <div class="topbar-badge">ENTERPRISE v2.0</div>
    <div class="topbar-badge topbar-badge-warn">{op_mode.upper()}</div>
  </div>
  <div class="topbar-status">
    <div class="status-item">
      <span class="sdot sdot-online"></span>
      <span class="status-label">SYSTEM</span>
      <span class="status-value">ONLINE</span>
    </div>
    <div style="width:1px;height:16px;background:rgba(37,99,235,0.12)"></div>
    <div class="status-item">
      <span class="sdot {_mode_dot}"></span>
      <span class="status-label">MODE</span>
      <span class="status-value">{_mode_lbl}</span>
    </div>
    <div style="width:1px;height:16px;background:rgba(37,99,235,0.12)"></div>
    <div class="status-item">
      <span class="sdot sdot-active"></span>
      <span class="status-label">CAM-01</span>
      <span class="status-value">LIVE</span>
    </div>
    <div style="width:1px;height:16px;background:rgba(37,99,235,0.12)"></div>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#A8B4CC">{_now_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not active_now:
    st.markdown('<div class="alert-warning"><span>🌙</span><span>System is in <b>Scheduled Sleep</b> mode — detection paused until the active window.</span></div>',unsafe_allow_html=True)

# ── METRIC BAR ────────────────────────────────────────────────────────────────
uptime = int(time.time()-st.session_state.start_time)
up_str = f"{uptime//3600:02d}:{(uptime%3600)//60:02d}:{uptime%60:02d}"

st.markdown(f"""
<div class="mgrid">
  <div class="mcard" style="--mc:#2563EB;--mb:#EFF6FF">
    <div class="mcard-icon">👤</div>
    <div class="mval">{st.session_state.people_count}</div>
    <div class="mlbl">People Detected</div>
    <div class="msub">this session</div>
  </div>
  <div class="mcard" style="--mc:#0891B2;--mb:#ECFEFF">
    <div class="mcard-icon">🚗</div>
    <div class="mval">{st.session_state.vehicle_count}</div>
    <div class="mlbl">Vehicles</div>
    <div class="msub">this session</div>
  </div>
  <div class="mcard" style="--mc:#DC2626;--mb:#FFF1F2">
    <div class="mcard-icon">🚨</div>
    <div class="mval">{st.session_state.alert_count}</div>
    <div class="mlbl">Alerts Triggered</div>
    <div class="msub">total count</div>
  </div>
  <div class="mcard" style="--mc:#D97706;--mb:#FFFBEB">
    <div class="mcard-icon">📸</div>
    <div class="mval">{st.session_state.snap_count}</div>
    <div class="mlbl">Snapshots</div>
    <div class="msub">auto-captured</div>
  </div>
  <div class="mcard" style="--mc:#059669;--mb:#ECFDF5">
    <div class="mcard-icon">⏱</div>
    <div class="mval" style="font-size:1.4rem;letter-spacing:-.5px">{up_str}</div>
    <div class="mlbl">System Uptime</div>
    <div class="msub">hh : mm : ss</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([13,9], gap="small")

with left_col:
    st.markdown("""
    <div class="panel">
      <div class="panel-hdr">
        <div class="panel-hdr-left">
          <div class="panel-hdr-dot"></div>
          <span class="panel-hdr-title">CAM-01 — Live Feed</span>
        </div>
        <span class="panel-badge panel-badge-live">● RECORDING</span>
      </div>
    """, unsafe_allow_html=True)

    facing = "environment" if "Back" in camera_facing else "user"
    ctx = webrtc_streamer(
        key=f"deepwatch-{facing}-{model_key}",
        video_processor_factory=AdvancedVideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video":{"facingMode":{"ideal":facing},"width":{"ideal":1280},
                     "height":{"ideal":720},"frameRate":{"ideal":30}},
            "audio":False,
        },
        async_processing=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="panel">
      <div class="panel-hdr">
        <div class="panel-hdr-left">
          <div class="panel-hdr-dot" style="background:#D97706;box-shadow:0 0 0 3px rgba(217,119,6,0.12)"></div>
          <span class="panel-hdr-title">Auto-Captured Snapshots</span>
        </div>
        <span class="panel-badge">{st.session_state.snap_count} TOTAL</span>
      </div>
    """, unsafe_allow_html=True)

    if st.session_state.snapshots:
        snap_list=list(st.session_state.snapshots)[:6]
        n_cols=min(3,len(snap_list)); cols=st.columns(n_cols)
        for idx,snap in enumerate(snap_list):
            with cols[idx%n_cols]:
                st.image(snap["img"],caption=f"{snap['label']} · {snap['time']}",use_container_width=True)
    else:
        st.markdown("""
        <div style="padding:18px;text-align:center;color:#A8B4CC;font-size:.74rem;
                    font-family:'IBM Plex Mono',monospace;background:#FAFBFE;
                    border-radius:8px;border:1px dashed #E8EAF4;line-height:1.7">
          No snapshots captured yet<br>
          <span style="font-size:.65rem;color:#C5CEDF">Detection events will trigger automatic capture</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if boundary_zones:
        st.markdown("""
        <div class="panel">
          <div class="panel-hdr">
            <div class="panel-hdr-left">
              <div class="panel-hdr-dot" style="background:#F59E0B;box-shadow:0 0 0 3px rgba(245,158,11,0.12)"></div>
              <span class="panel-hdr-title">Boundary Zone Status</span>
            </div>
          </div>
        """, unsafe_allow_html=True)
        for y_pct,z_label,z_hex in boundary_zones:
            st.markdown(
                f'<div class="zone-badge"><div class="zone-dot" style="background:{z_hex}"></div>'
                f'<span class="zone-text">{z_label} — active boundary at {y_pct}% frame height</span></div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


with right_col:
    st.markdown("""
    <div class="panel">
      <div class="panel-hdr">
        <div class="panel-hdr-left">
          <div class="panel-hdr-dot" style="background:#DC2626;box-shadow:0 0 0 3px rgba(220,38,38,0.12)"></div>
          <span class="panel-hdr-title">Live Detections</span>
        </div>
        <span class="panel-badge panel-badge-live">LIVE</span>
      </div>
    """, unsafe_allow_html=True)

    alert_placeholder  = st.empty()
    detect_placeholder = st.empty()

    if ctx.video_processor:
        proc = ctx.video_processor
        with proc._lock:
            dets          = list(proc.detections)
            fps_val       = proc.fps
            last_frm      = proc.last_frame.copy() if proc.last_frame is not None else None
            zone_breaches = getattr(proc,"_zone_breaches",[])
            loiter_alerts = getattr(proc,"_loiter_alerts",[])
            n_people_live = getattr(proc,"_n_people",0)

        if dets:
            for d in dets:
                st.session_state.label_counts[d["label"]]+=1
                if d["category"]=="person": st.session_state.people_count+=1
                elif d["category"]=="vehicle": st.session_state.vehicle_count+=1
            st.session_state.total_detections+=len(dets)

        crowd_alert=""
        if ac_crowd and n_people_live>=crowd_n:
            crowd_alert=f"CROWD DENSITY ALERT — {n_people_live} PEOPLE IN FRAME"

        weapon_dets=[d for d in dets if d["category"]=="weapon"]
        now_t2=time.time()

        if weapon_dets and last_frm is not None:
            if now_t2-st.session_state.get("last_weapon_alert",0)>10:
                st.session_state["last_weapon_alert"]=now_t2
                st.session_state.alert_count+=1
                weapon_labels=", ".join(f"{d['label'].upper()} ({int(d['conf']*100)}%)" for d in weapon_dets)
                alert_placeholder.markdown(f"""
                <div class="alert-critical">
                  <div class="alert-critical-icon">⚠️</div>
                  <div>
                    <div class="alert-critical-title">WEAPON DETECTED — IMMEDIATE RESPONSE REQUIRED</div>
                    <div class="alert-critical-sub">{weapon_labels}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                st.session_state.event_log.appendleft({"time":datetime.datetime.now().strftime("%H:%M:%S"),
                    "msg":f"⚠ WEAPON — {weapon_labels}","level":"critical","count":len(weapon_dets)})
                threading.Thread(target=db_insert,
                    args=("critical","weapon",weapon_labels,f"WEAPON DETECTED: {weapon_labels}",
                          max(d["conf"] for d in weapon_dets),"CAM-01"),daemon=True).start()

                _,buf=cv2.imencode(".jpg",last_frm,[cv2.IMWRITE_JPEG_QUALITY,95])
                snap_bytes=buf.tobytes()
                snap_rgb=cv2.cvtColor(last_frm,cv2.COLOR_BGR2RGB)
                st.session_state.snapshots.appendleft({"img":snap_rgb,
                    "time":datetime.datetime.now().strftime("%H:%M:%S"),
                    "label":f"⚠ {weapon_labels}","bytes":snap_bytes})
                st.session_state.snap_count+=1; st.session_state.last_snap_time=now_t2

                if email_enabled:
                    max_conf_w=max(d["conf"] for d in weapon_dets)
                    _wsubj="WEAPON DETECTED on CAM-01 — IMMEDIATE ACTION REQUIRED"
                    _wbody=(f"⚠  WEAPON ALERT — DeepWatch Enterprise\n{'='*50}\n\n"
                             f"Weapon(s): {weapon_labels}\nConfidence: {max_conf_w:.0%}\n"
                             f"Camera: CAM-01\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                             f"{'='*50}\nSnapshot attached. Immediate action recommended.\n")
                    def _send_weapon(subj,body,img_b):
                        ok,msg=send_email_alert(subj,body,img_b)
                        st.session_state.email_count+=1
                        st.session_state.event_log.appendleft({"time":datetime.datetime.now().strftime("%H:%M:%S"),
                            "msg":f"Email {'✓ sent' if ok else '✗ '+msg}","level":"info" if ok else "critical","count":0})
                    threading.Thread(target=_send_weapon,args=(_wsubj,_wbody,snap_bytes),daemon=True).start()

        hit_cats={d["category"] for d in dets}&st.session_state.alert_cats
        all_alerts=[]
        if hit_cats and "weapon" not in hit_cats: all_alerts+=[f"DETECTED — {', '.join(sorted(hit_cats)).upper()}"]
        if zone_breaches: all_alerts+=zone_breaches
        if loiter_alerts: all_alerts+=loiter_alerts
        if crowd_alert: all_alerts.append(crowd_alert)

        if all_alerts and (now_t2-st.session_state.last_alert_time>alert_cooldown):
            st.session_state.last_alert_time=now_t2
            st.session_state.alert_count+=1
            alert_html="".join(f"""<div class="alert-critical">
              <div class="alert-critical-icon">🚨</div>
              <div><div class="alert-critical-title">{a}</div></div>
            </div>""" for a in all_alerts)
            if not weapon_dets: alert_placeholder.markdown(alert_html,unsafe_allow_html=True)

            for a in all_alerts:
                st.session_state.event_log.appendleft({"time":datetime.datetime.now().strftime("%H:%M:%S"),
                    "msg":a,"level":"critical","count":len(dets)})
                threading.Thread(target=db_insert,args=("critical","alert","ALERT",a,1.0,"CAM-01"),daemon=True).start()

            snap_hit={d["category"] for d in dets}&st.session_state.snap_cats
            snap_hit.discard("weapon")
            if snap_hit and last_frm is not None:
                if now_t2-st.session_state.last_snap_time>snap_cooldown:
                    st.session_state.last_snap_time=now_t2; st.session_state.snap_count+=1
                    _,buf=cv2.imencode(".jpg",last_frm); snap_bytes=buf.tobytes()
                    snap_rgb=cv2.cvtColor(last_frm,cv2.COLOR_BGR2RGB)
                    hit_labels=", ".join(d["label"].upper() for d in dets if d["category"] in snap_hit)
                    st.session_state.snapshots.appendleft({"img":snap_rgb,
                        "time":datetime.datetime.now().strftime("%H:%M:%S"),
                        "label":hit_labels,"bytes":snap_bytes})
                    if email_enabled:
                        max_conf=max((d["conf"] for d in dets if d["category"] in snap_hit),default=0)
                        email_subj=f"{hit_labels} detected on CAM-01"
                        email_body=(f"Detection: {hit_labels}\nConfidence: {max_conf:.0%}\n"
                            f"Zone breaches: {', '.join(zone_breaches) or 'None'}\n"
                            f"Loitering: {', '.join(loiter_alerts) or 'None'}\n"
                            f"Time: {datetime.datetime.now().isoformat()}\nCamera: CAM-01\n")
                        def _send(subj,body,img_b):
                            ok,msg=send_email_alert(subj,body,img_b)
                            st.session_state.email_count+=1
                            st.session_state.event_log.appendleft({"time":datetime.datetime.now().strftime("%H:%M:%S"),
                                "msg":f"Email {'✓ sent' if ok else '✗ '+msg}","level":"info" if ok else "critical","count":0})
                        threading.Thread(target=_send,args=(email_subj,email_body,snap_bytes),daemon=True).start()

        if dets:
            tags_html="".join(
                f'<span class="otag t-{d["category"]}">{CAT_ICON.get(d["category"],"📦")} {d["label"]} {int(d["conf"]*100)}%</span>'
                for d in sorted(dets,key=lambda x:-x["conf"]))
            bars_html=""
            for d in sorted(dets,key=lambda x:-x["conf"])[:10]:
                pct=int(d["conf"]*100); hx=CAT_HEX.get(d["category"],"#2563EB")
                bars_html+=f"""
                <div class="det-item">
                  <div class="det-row">
                    <span class="det-label">{CAT_ICON.get(d["category"],"📦")} {d["label"].upper()}</span>
                    <span class="det-conf" style="color:{hx}">{pct}%</span>
                  </div>
                  <div class="cbar-wrap">
                    <div class="cbar" style="width:{pct}%;background:linear-gradient(90deg,{hx},{hx}88)"></div>
                  </div>
                </div>"""
            fps_html=f'<div class="fps-strip"><span class="fps-chip">FPS {fps_val}</span><span class="fps-chip">FRAME {proc.frame_count}</span><span class="fps-chip">OBJ {len(dets)}</span></div>'
            detect_placeholder.markdown(tags_html+"<br>"+bars_html+fps_html,unsafe_allow_html=True)
        else:
            detect_placeholder.markdown("""
            <div style="padding:16px;text-align:center;color:#A8B4CC;font-size:.73rem;
                        font-family:'IBM Plex Mono',monospace;background:#FAFBFE;
                        border-radius:8px;border:1px dashed #E8EAF4">
              No objects detected in current frame
            </div>""", unsafe_allow_html=True)
    else:
        detect_placeholder.markdown("""
        <div style="padding:16px;text-align:center;color:#A8B4CC;font-size:.73rem;
                    font-family:'IBM Plex Mono',monospace;background:#FAFBFE;
                    border-radius:8px;border:1px dashed #E8EAF4">
          Press <b>Start</b> to activate the camera feed
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Statistics
    st.markdown(f"""
    <div class="panel">
      <div class="panel-hdr">
        <div class="panel-hdr-left">
          <div class="panel-hdr-dot" style="background:#059669;box-shadow:0 0 0 3px rgba(5,150,105,0.12)"></div>
          <span class="panel-hdr-title">Session Statistics</span>
        </div>
        <span class="panel-badge">{st.session_state.total_detections:,} TOTAL</span>
      </div>
    """, unsafe_allow_html=True)

    if st.session_state.label_counts:
        top_items=sorted(st.session_state.label_counts.items(),key=lambda x:-x[1])[:10]
        max_c=max(c for _,c in top_items); bars_s=""
        for label,count in top_items:
            cid_m=next((i for i,n in model_obj.names.items() if n==label),999)
            cat=get_category(cid_m); hx=CAT_HEX.get(cat,"#2563EB"); bw=int(count/max_c*100)
            bars_s+=f"""
            <div class="stat-item">
              <div class="stat-row">
                <span class="stat-label">{CAT_ICON.get(cat,"📦")} {label.upper()}</span>
                <span class="stat-count" style="color:{hx}">{count:,}×</span>
              </div>
              <div class="cbar-wrap">
                <div class="cbar" style="width:{bw}%;background:linear-gradient(90deg,{hx},{hx}55)"></div>
              </div>
            </div>"""
        st.markdown(bars_s, unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:12px 8px;color:#A8B4CC;font-size:.73rem;font-family:\'IBM Plex Mono\',monospace;text-align:center">Awaiting first detection…</div>',unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Event Log
    st.markdown("""
    <div class="panel">
      <div class="panel-hdr">
        <div class="panel-hdr-left">
          <div class="panel-hdr-dot" style="background:#8B5CF6;box-shadow:0 0 0 3px rgba(139,92,246,0.12)"></div>
          <span class="panel-hdr-title">Event Log</span>
        </div>
        <span class="panel-badge panel-badge-live">LIVE</span>
      </div>
    """, unsafe_allow_html=True)

    log_tab1,log_tab2=st.tabs(["Live (Session)","Persisted (DB)"])

    with log_tab1:
        if st.session_state.event_log:
            lvl_map={"critical":"lc","warning":"lw","info":"li","success":"ls"}
            html=""
            for e in list(st.session_state.event_log)[:20]:
                cls=lvl_map.get(e.get("level","info"),"li")
                cnt=f"+{e['count']}" if e.get("count") else ""
                msg=e["msg"][:58]+("…" if len(e["msg"])>58 else "")
                html+=f'<div class="log-row {cls}"><span class="log-t">{e["time"]}</span><span class="log-m">{msg}</span><span class="log-n">{cnt}</span></div>'
            st.markdown(html,unsafe_allow_html=True)
        else:
            st.markdown('<div style="padding:12px;color:#A8B4CC;font-size:.7rem;font-family:\'IBM Plex Mono\',monospace;text-align:center">No events recorded yet</div>',unsafe_allow_html=True)

    with log_tab2:
        db_rows=db_fetch_recent(30)
        if db_rows:
            lvl_map={"critical":"lc","warning":"lw","info":"li"}; html=""
            for row in db_rows:
                ts,level,cat,label,msg,conf,cam=row
                cls=lvl_map.get(level,"li"); t_s=ts[11:19]; conf_s=f"{conf:.0%}" if conf else ""
                html+=f'<div class="log-row {cls}"><span class="log-t">{t_s}</span><span class="log-m">[{cam}] {msg[:45]}</span><span class="log-n" style="color:#7B8DB0">{conf_s}</span></div>'
            st.markdown(html,unsafe_allow_html=True)
        else:
            st.markdown('<div style="padding:12px;color:#A8B4CC;font-size:.7rem;font-family:\'IBM Plex Mono\',monospace;text-align:center">No persisted events yet</div>',unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vfooter">
  <strong>DeepWatch Enterprise v2.0</strong>
  &nbsp;·&nbsp; YOLOv8 &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp;
  OpenCV &nbsp;·&nbsp; WebRTC &nbsp;·&nbsp; SQLite
  &nbsp;&nbsp;|&nbsp;&nbsp;
  EU AI Act Compliant &nbsp;·&nbsp; Educational Portfolio Project &nbsp;·&nbsp;
  2026 &nbsp;·&nbsp; Created by Raja Roy
</div>
""", unsafe_allow_html=True)

# streamlit run app_version2.py