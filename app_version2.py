"""
DeepWatch v1.0 — Advanced CCTV Control Room
============================================
Full-featured AI surveillance dashboard with:
  • YOLOv8 real-time object detection
  • Multi-camera support (switchable front/rear)
  • Virtual boundary / tripwire zones (configurable)
  • Person tracking with unique IDs (ByteTrack)
  • Loitering detection (time-in-zone)
  • Crowd density estimation
  • Heatmap accumulation
  • Auto-snapshots with annotated overlay
  • Email alerts with snapshot attachment
  • Scheduled / Vacation / Always-On modes
  • SQLite event persistence across sessions
  • CSV / JSON export of event log
  • Streamlit real-time metric updates via st.empty()
  • Full dark cyberpunk UI (Share Tech Mono + Exo 2)

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
    page_title="DeepWatch — AI CCTV Control Room",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "DeepWatch v1.0 — Deep Learning Meets Real-Time Security"},
)

# ─────────────────────────────── CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:ital,wght@0,300;0,400;0,600;0,700;0,900;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #04060f;
    color: #b8cce0;
    font-family: 'Exo 2', sans-serif;
}

/* ── Animated grid background ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: -2;
    background-image:
        linear-gradient(rgba(0,180,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,180,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}
.stApp::after {
    content: '';
    position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background:
        radial-gradient(ellipse 80% 60% at 0% 0%,   rgba(0,40,80,0.6) 0%, transparent 50%),
        radial-gradient(ellipse 60% 50% at 100% 100%, rgba(0,10,40,0.8) 0%, transparent 50%),
        radial-gradient(ellipse 40% 30% at 50% 50%,  rgba(0,180,255,0.04) 0%, transparent 70%);
}

/* ── Scanlines overlay ── */
body::after {
    content: '';
    position: fixed; inset: 0; z-index: 9998; pointer-events: none;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 3px,
        rgba(0,0,0,0.04) 3px, rgba(0,0,0,0.04) 4px
    );
}

/* ── Layout ── */
.main .block-container { padding: 0.5rem 1rem 1rem; max-width: 100%; }
section[data-testid="stSidebar"] {
    background: rgba(2,6,18,0.97) !important;
    border-right: 1px solid rgba(0,180,255,0.12) !important;
    width: 280px !important;
}
section[data-testid="stSidebar"] * { color: #b8cce0 !important; }
section[data-testid="stSidebar"] label { font-size: .78rem !important; }
section[data-testid="stSidebar"] .stSlider { padding-bottom: 6px !important; }

/* ── Top Bar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(90deg, rgba(0,20,50,0.95), rgba(0,10,30,0.95));
    border: 1px solid rgba(0,180,255,0.25);
    border-radius: 10px; padding: 10px 22px; margin-bottom: 10px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 30px rgba(0,180,255,0.08), inset 0 1px 0 rgba(255,255,255,0.04);
}
.topbar-left { display: flex; align-items: center; gap: 18px; }
.topbar-logo {
    font-family: 'Share Tech Mono', monospace; font-size: 1.25rem;
    color: #00c8ff; letter-spacing: 4px;
    text-shadow: 0 0 20px rgba(0,200,255,0.6), 0 0 40px rgba(0,200,255,0.2);
}
.topbar-logo em { color: #ff3e3e; font-style: normal; }
.topbar-badge {
    background: rgba(0,200,255,0.1); border: 1px solid rgba(0,200,255,0.3);
    border-radius: 4px; padding: 2px 10px;
    font-family: 'Share Tech Mono', monospace; font-size: .65rem;
    color: #00c8ff; letter-spacing: 2px;
}
.topbar-status {
    display: flex; gap: 16px; align-items: center;
    font-family: 'Share Tech Mono', monospace; font-size: .72rem; color: #456;
}
.sdot {
    width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 5px;
}
.sdot-g { background:#00ff88; box-shadow:0 0 8px #00ff88; animation: sblink 2s ease infinite; }
.sdot-r { background:#ff3030; box-shadow:0 0 8px #ff3030; animation: sblink .8s ease infinite; }
.sdot-y { background:#ffc400; box-shadow:0 0 8px #ffc400; animation: sblink 1.5s ease infinite; }
.sdot-b { background:#00c8ff; box-shadow:0 0 8px #00c8ff; }
@keyframes sblink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Panels ── */
.panel {
    background: rgba(0,12,28,0.88);
    border: 1px solid rgba(0,180,255,0.14);
    border-radius: 10px; padding: 14px;
    backdrop-filter: blur(10px);
    margin-bottom: 10px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
}
.panel-hdr {
    font-family: 'Share Tech Mono', monospace; font-size: .7rem;
    color: #00c8ff; letter-spacing: 2.5px; text-transform: uppercase;
    margin-bottom: 10px; padding-bottom: 8px;
    border-bottom: 1px solid rgba(0,180,255,0.1);
    display: flex; align-items: center; justify-content: space-between;
}
.panel-hdr-badge {
    background: rgba(0,200,255,0.1); border: 1px solid rgba(0,200,255,0.25);
    border-radius: 3px; padding: 1px 8px; font-size: .62rem; color: #00c8ff;
}

/* ── Metric Grid ── */
.mgrid { display: grid; grid-template-columns: repeat(5,1fr); gap: 8px; margin-bottom: 10px; }
.mcard {
    background: rgba(0,15,35,0.92);
    border: 1px solid rgba(0,180,255,0.18);
    border-radius: 8px; padding: 11px 13px;
    position: relative; overflow: hidden;
    transition: border-color .3s;
}
.mcard::before {
    content: ''; position: absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, var(--ac) 0%, transparent 100%);
}
.mcard::after {
    content: ''; position: absolute; bottom:0; right:0;
    width: 50px; height: 50px; border-radius: 50%;
    background: radial-gradient(circle, var(--ac) 0%, transparent 70%);
    opacity: .06;
}
.mval {
    font-family: 'Share Tech Mono', monospace; font-size: 1.85rem; font-weight: 700;
    color: var(--ac); text-shadow: 0 0 15px var(--ac); line-height: 1;
}
.mlbl { font-size: .63rem; color: #4a6075; letter-spacing: 1.5px;
        text-transform: uppercase; margin-top: 4px; }
.msub { font-size: .68rem; color: #5a7090; margin-top: 2px;
        font-family: 'Share Tech Mono', monospace; }
.mdelta {
    font-family: 'Share Tech Mono', monospace; font-size: .62rem;
    position: absolute; top: 8px; right: 10px;
}
.mdelta-up { color: #00ff88; }
.mdelta-dn { color: #ff4b4b; }

/* ── Alert Strip ── */
.alert-crit {
    background: linear-gradient(135deg, rgba(255,30,30,.2), rgba(220,60,0,.15));
    border: 1px solid rgba(255,50,50,.6); border-radius: 7px;
    padding: 9px 14px; margin: 5px 0;
    font-family: 'Share Tech Mono', monospace; font-size: .8rem; color: #ff6060;
    display: flex; align-items: center; gap: 10px;
    animation: apulse .7s ease infinite alternate;
}
.alert-warn {
    background: rgba(255,196,0,.08); border: 1px solid rgba(255,196,0,.35);
    border-radius: 7px; padding: 8px 14px; margin: 4px 0;
    font-family: 'Share Tech Mono', monospace; font-size: .76rem; color: #ffc400;
}
.alert-info {
    background: rgba(0,180,255,.07); border: 1px solid rgba(0,180,255,.25);
    border-radius: 7px; padding: 7px 12px; margin: 3px 0;
    font-family: 'Share Tech Mono', monospace; font-size: .73rem; color: #00a8dd;
}
@keyframes apulse {
    from { box-shadow: 0 0 8px rgba(255,40,40,.2); }
    to   { box-shadow: 0 0 22px rgba(255,40,40,.6); border-color: #ff7070; }
}

/* ── Object Tags ── */
.otag {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 9px; border-radius: 4px; border: 1px solid;
    font-family: 'Share Tech Mono', monospace; font-size: .69rem; margin: 2px;
}
.t-person  { color:#ff6b6b; border-color:rgba(255,107,107,.4); background:rgba(255,107,107,.08); }
.t-vehicle { color:#00d4ff; border-color:rgba(0,212,255,.4);   background:rgba(0,212,255,.08); }
.t-animal  { color:#48ff80; border-color:rgba(72,255,128,.4);  background:rgba(72,255,128,.08); }
.t-object  { color:#ffc400; border-color:rgba(255,196,0,.4);   background:rgba(255,196,0,.08); }

/* ── Progress / Confidence bars ── */
.cbar-wrap { background:rgba(255,255,255,.05); border-radius:2px; height:3px; margin-top:3px; }
.cbar { height:3px; border-radius:2px; transition: width .4s ease; }

/* ── Event log ── */
.log-row {
    display: flex; gap: 8px; align-items: center;
    padding: 5px 9px; border-radius: 5px; border-left: 2px solid;
    margin: 2px 0; font-family: 'Share Tech Mono', monospace; font-size: .69rem;
    background: rgba(0,15,35,.5);
}
.log-t { color: #3a5a70; min-width: 62px; }
.log-m { flex: 1; }
.log-n { color: #00c8ff; min-width: 28px; text-align: right; }
.lc { border-color: #ff4b4b; }
.lw { border-color: #ffc400; }
.li { border-color: #00c8ff; }
.ls { border-color: #00ff88; }

/* ── Zone box ── */
.zone-info {
    background: rgba(255,196,0,.06); border: 1px solid rgba(255,196,0,.25);
    border-radius: 6px; padding: 8px 12px; margin: 5px 0;
    font-family: 'Share Tech Mono', monospace; font-size: .72rem; color: #ffc400;
}

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg,#002850,#004070) !important;
    color: #00c8ff !important; border: 1px solid rgba(0,180,255,.4) !important;
    border-radius: 6px !important; font-family: 'Share Tech Mono',monospace !important;
    font-size: .75rem !important; letter-spacing: 1px !important;
    transition: all .2s !important; padding: 6px 14px !important;
}
.stButton button:hover {
    background: linear-gradient(135deg,#003870,#006090) !important;
    box-shadow: 0 0 16px rgba(0,180,255,.35) !important;
    border-color: rgba(0,200,255,.7) !important;
}

/* ── Sidebar widgets ── */
.stCheckbox label { font-size:.76rem !important; }
.stSelectbox label, .stRadio label { font-size:.76rem !important; }
div[data-baseweb="slider"] { margin-top: -6px !important; }

/* ── Footer ── */
.vfooter {
    text-align: center; padding: .6rem; margin-top: 8px;
    color: #1e3050; font-family: 'Share Tech Mono', monospace;
    font-size: .65rem; letter-spacing: 1.5px;
    border-top: 1px solid rgba(0,180,255,0.07);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: rgba(0,10,25,1); }
::-webkit-scrollbar-thumb { background: rgba(0,180,255,.25); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,180,255,.5); }

hr { border-color: rgba(0,180,255,0.08) !important; margin: 8px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────── DATABASE (SQLite) ────────────────────────────────────
DB_PATH = Path("deepwatch_events.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT,
            level     TEXT,
            category  TEXT,
            label     TEXT,
            message   TEXT,
            conf      REAL,
            cam       TEXT
        )
    """)
    con.commit()
    con.close()

def db_insert(level, category, label, message, conf=0.0, cam="CAM-01"):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO events (ts,level,category,label,message,conf,cam) VALUES (?,?,?,?,?,?,?)",
            (datetime.datetime.now().isoformat(), level, category, label, message, conf, cam)
        )
        con.commit()
        con.close()
    except Exception:
        pass

def db_fetch_recent(n=100):
    try:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT ts,level,category,label,message,conf,cam FROM events ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        con.close()
        return rows
    except Exception:
        return []

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
    try:
        return st.secrets.get(k, os.getenv(k, d))
    except Exception:
        return os.getenv(k, d)

_turn_urls = [u for u in [_env("TURN_URL_1"), _env("TURN_URL_2"),
                            _env("TURN_URL_3"), _env("TURN_URL_4")] if u.strip()]
_tu = _env("TURN_USERNAME")
_tp = _env("TURN_PASSWORD")

if _turn_urls and _tu and _tp:
    _ice = [
        {"urls": ["stun:stun.relay.metered.ca:80"]},
        {"urls": _turn_urls, "username": _tu, "credential": _tp},
    ]
else:
    _ice = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["turn:openrelay.metered.ca:80",
                  "turn:openrelay.metered.ca:443",
                  "turn:openrelay.metered.ca:443?transport=tcp"],
         "username": "openrelayproject", "credential": "openrelayproject"},
    ]

_rtc_cfg_dict = {"iceServers": _ice}
if _env("FORCE_TURN", "false").lower() == "true":
    _rtc_cfg_dict["iceTransportPolicy"] = "relay"

RTC_CONFIG = RTCConfiguration(_rtc_cfg_dict)


# ─────────────────────────── YOLO MODEL ──────────────────────────────────────
@st.cache_resource
def load_model(weights: str = "yolov8n.pt"):
    from pathlib import Path
    if not Path(weights).exists() and weights not in ["yolov8n.pt","yolov8s.pt","yolov8m.pt"]:
        return None   # model file missing — handled gracefully in UI
    return YOLO(weights)

# ── Model registry ─────────────────────────────────────────────────────────
# "is_weapon_model" = True  →  every class the model detects is treated as WEAPON
# "file"           = filename to look for in the project folder
MODEL_OPTIONS = {
    # ── Standard COCO models ──────────────────────────────────────────────
    "YOLOv8n · General (fastest)":  {
        "file": "yolov8n.pt", "is_weapon_model": False,
        "desc": "80 COCO classes · people, cars, animals…"
    },
    "YOLOv8s · General (balanced)": {
        "file": "yolov8s.pt", "is_weapon_model": False,
        "desc": "80 COCO classes · better accuracy"
    },
    "YOLOv8m · General (accurate)": {
        "file": "yolov8m.pt", "is_weapon_model": False,
        "desc": "80 COCO classes · highest accuracy"
    },
    # ── Custom Weapon models (place .pt files in DeepWatch/ folder) ───────
    "🔫 WeaponV1 · 14 Classes":     {
        "file": "weapon_v1.pt", "is_weapon_model": True,
        "desc": "AK47 · Rifle · Revolver · Shotgun · Knife · Axe · Sword · M16…"
    },
    "🔫 WeaponV2 · 5 Classes":      {
        "file": "weapon_v2.pt", "is_weapon_model": True,
        "desc": "Pistol · Rifle · Knife · Grenade · Missile"
    },
    "🔫 WeaponV3 · Gun+Knife":      {
        "file": "weapon_v3.pt", "is_weapon_model": True,
        "desc": "Handgun · Shotgun · Knife · Rifle"
    },
}

# ── Weapon keyword matcher (for custom model label names) ─────────────────
WEAPON_KEYWORDS = {
    "gun","pistol","rifle","revolver","shotgun","ak47","m16","firearm",
    "weapon","knife","sword","axe","grenade","missile","handgun","sniper",
    "carbine","uzi","glock","assault","explosive","bomb","blade",
}

# ── COCO class mappings (used only for general models) ────────────────────
PERSON_IDS  = {0}
VEHICLE_IDS = {1, 2, 3, 5, 7}
ANIMAL_IDS  = set(range(14, 24))
WEAPON_IDS  = {76, 43}                  # scissors, knife in COCO

CAT_BGR  = {
    "person":  (80,  80,  255),
    "vehicle": (255, 200, 0),
    "animal":  (80,  255, 128),
    "object":  (0,   200, 255),
    "weapon":  (0,   0,   255),
}
CAT_HEX  = {
    "person":  "#ff6b6b",
    "vehicle": "#00d4ff",
    "animal":  "#48ff80",
    "object":  "#ffc400",
    "weapon":  "#ff3030",
}
CAT_ICON = {
    "person":  "👤",
    "vehicle": "🚗",
    "animal":  "🐾",
    "object":  "📦",
    "weapon":  "⚠️",
}

def get_category(cid: int, label: str = "", is_weapon_model: bool = False) -> str:
    """Smart category detection — handles both COCO and custom weapon models."""
    # If this is a dedicated weapon model → everything is a weapon
    if is_weapon_model:
        return "weapon"
    # Check label name for weapon keywords (catches edge cases)
    if label.lower() in WEAPON_KEYWORDS:
        return "weapon"
    # Standard COCO ID checks
    if cid in PERSON_IDS:  return "person"
    if cid in VEHICLE_IDS: return "vehicle"
    if cid in ANIMAL_IDS:  return "animal"
    if cid in WEAPON_IDS:  return "weapon"
    return "object"


# ─────────────────────────── EMAIL HELPER ────────────────────────────────────
def send_email_alert(subject: str, body: str, img_bytes: bytes = None) -> tuple[bool, str]:
    smtp_host = _env("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(_env("SMTP_PORT", "587"))
    smtp_user = _env("SMTP_USER", "")
    smtp_pass = _env("SMTP_PASS", "")
    smtp_to   = _env("SMTP_TO",   smtp_user)

    if not smtp_user or not smtp_pass:
        return False, "SMTP credentials not configured"

    try:
        msg = MIMEMultipart()
        msg["From"]    = smtp_user
        msg["To"]      = smtp_to
        msg["Subject"] = f"[DeepWatch 🚨] {subject}"
        body_full = (
            f"{body}\n\n"
            f"────────────────────────────\n"
            f"DeepWatch v1.0 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"This is an automated security alert.\n"
        )
        msg.attach(MIMEText(body_full, "plain"))
        if img_bytes:
            att = MIMEImage(img_bytes, name="deepwatch_alert.jpg")
            att.add_header("Content-Disposition", "attachment", filename="deepwatch_alert.jpg")
            msg.attach(att)
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as srv:
            srv.starttls()
            srv.login(smtp_user, smtp_pass)
            srv.send_message(msg)
        return True, "OK"
    except Exception as exc:
        return False, str(exc)


# ─────────────────────── SESSION STATE INIT ──────────────────────────────────
_defaults: dict = {
    # counters
    "total_detections":  0,
    "people_count":      0,
    "vehicle_count":     0,
    "alert_count":       0,
    "snap_count":        0,
    "email_count":       0,
    # per-label counts
    "label_counts":      defaultdict(int),
    # in-memory log (last 300)
    "event_log":         deque(maxlen=300),
    # snapshots (last 12)
    "snapshots":         deque(maxlen=12),
    # heatmap accumulator (updated by processor)
    "heatmap":           None,
    # loitering: track_id → entry_time
    "loiter_times":      {},
    # timing
    "start_time":        time.time(),
    "last_alert_time":   0.0,
    "last_snap_time":    0.0,
    "last_weapon_alert": 0.0,
    # config (overwritten by sidebar)
    "alert_cats":        {"person"},
    "snap_cats":         {"person"},
    "boundary_zones":    [],     # list of (y_pct, label, color_hex)
    "loiter_threshold":  8,
    "model_key":         "YOLOv8n (fastest)",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────── SIDEBAR ─────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Share Tech Mono\',monospace;color:#00c8ff;'
        'letter-spacing:3px;font-size:.85rem;padding:6px 0 2px">⚙  CONTROL PANEL</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Camera ──
    st.markdown("**📷 Camera**")
    camera_facing = st.selectbox("Feed", ["Back (Rear)", "Front (Selfie)"], index=0,
                                  label_visibility="collapsed")
    model_key = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
    st.session_state.model_key = model_key

    # Show model description
    _minfo = MODEL_OPTIONS[model_key]
    st.markdown(
        f'<div style="background:rgba(0,180,255,.06);border:1px solid rgba(0,180,255,.15);'
        f'border-radius:5px;padding:6px 10px;font-family:\'Share Tech Mono\',monospace;'
        f'font-size:.63rem;color:#4a7090;line-height:1.5">'
        f'{"⚠️ WEAPON MODEL" if _minfo["is_weapon_model"] else "🌐 GENERAL MODEL"}<br>'
        f'{_minfo["desc"]}</div>',
        unsafe_allow_html=True
    )

    # Check if weapon model file exists
    _model_file = _minfo["file"]
    _model_missing = (_minfo["is_weapon_model"] and not Path(_model_file).exists())
    if _model_missing:
        st.markdown(
            f'<div style="background:rgba(255,100,0,.1);border:1px solid rgba(255,100,0,.4);'
            f'border-radius:5px;padding:6px 10px;font-family:\'Share Tech Mono\',monospace;'
            f'font-size:.63rem;color:#ff8040;margin-top:4px">'
            f'⚠️ FILE NOT FOUND: <b>{_model_file}</b> — place in DeepWatch/ folder<br>'
            f'Falling back to YOLOv8n general model</div>',
            unsafe_allow_html=True
        )
        model_obj = load_model("yolov8n.pt")
    else:
        model_obj = load_model(_model_file)

    IS_WEAPON_MODEL = _minfo["is_weapon_model"] and not _model_missing

    conf_thresh = st.slider("Detection Confidence", 0.20, 0.90, 0.45, 0.05,
                             format="%.2f")
    max_det = st.slider("Max Detections / Frame", 5, 50, 20, 5)

    st.markdown("---")

    # ── Operation mode ──
    st.markdown("**🕐 Operation Mode**")
    op_mode = st.radio("Mode", ["Always On", "Vacation (24/7)", "Scheduled"],
                        label_visibility="collapsed")
    active_now = True
    if op_mode == "Scheduled":
        col1, col2 = st.columns(2)
        with col1:
            t_start = st.time_input("Start", datetime.time(22, 0))
        with col2:
            t_end   = st.time_input("End",   datetime.time(6, 0))
        now_t = datetime.datetime.now().time()
        if t_start < t_end:
            active_now = t_start <= now_t <= t_end
        else:
            active_now = now_t >= t_start or now_t <= t_end

    st.markdown("---")

    # ── Boundary Zones ──
    st.markdown("**🚧 Boundary Zones**")
    bz_count = st.number_input("Number of zones", 0, 4, 1, 1)
    boundary_zones = []
    zone_colors = ["#ffc400", "#ff4b4b", "#00ff88", "#00c8ff"]
    zone_names  = ["ZONE-A", "ZONE-B", "ZONE-C", "ZONE-D"]
    for i in range(int(bz_count)):
        y_pct = st.slider(f"{zone_names[i]} height (%)", 0, 100, 65 - i*15, 5,
                           key=f"zone_{i}")
        boundary_zones.append((y_pct, zone_names[i], zone_colors[i]))
    st.session_state.boundary_zones = boundary_zones

    st.markdown("---")

    # ── Alert triggers ──
    st.markdown("**🚨 Alert Triggers**")
    ac_person  = st.checkbox("Person",  value=True,  key="ac_p")
    ac_vehicle = st.checkbox("Vehicle", value=False, key="ac_v")
    ac_animal  = st.checkbox("Animal",  value=False, key="ac_a")
    ac_object  = st.checkbox("Object",  value=False, key="ac_o")
    ac_weapon  = st.checkbox("Weapon ⚠", value=True, key="ac_w")
    ac_loiter  = st.checkbox("Loitering", value=True, key="ac_l")
    ac_crowd   = st.checkbox("Crowd (>N people)", value=False, key="ac_cr")
    crowd_n    = st.slider("Crowd threshold", 2, 20, 5, 1) if ac_crowd else 5

    alert_cats = set()
    if ac_person:  alert_cats.add("person")
    if ac_vehicle: alert_cats.add("vehicle")
    if ac_animal:  alert_cats.add("animal")
    if ac_object:  alert_cats.add("object")
    if ac_weapon:  alert_cats.add("weapon")
    st.session_state.alert_cats = alert_cats

    alert_cooldown = st.slider("Alert cooldown (s)", 5, 120, 30, 5)

    st.markdown("---")

    # ── Loitering ──
    st.markdown("**⏳ Loitering Detection**")
    loiter_thresh = st.slider("Loiter threshold (s)", 3, 60, 8, 1)
    st.session_state.loiter_threshold = loiter_thresh

    st.markdown("---")

    # ── Snapshot ──
    st.markdown("**📸 Auto Snapshot**")
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
    snap_cooldown = st.slider("Snap cooldown (s)", 2, 60, 10, 2)

    st.markdown("---")

    # ── Email ──
    st.markdown("**📧 Email Alerts**")
    email_enabled = st.checkbox("Enable email alerts", value=False)
    if email_enabled:
        st.markdown(
            '<div style="background:rgba(0,180,255,.06);border:1px solid rgba(0,180,255,.2);'
            'border-radius:5px;padding:7px 10px;font-family:\'Share Tech Mono\',monospace;'
            'font-size:.65rem;color:#456;line-height:1.6">'
            'Set in Streamlit Secrets:<br>'
            '<span style="color:#00c8ff">SMTP_USER SMTP_PASS SMTP_TO</span><br>'
            'Optional: SMTP_HOST SMTP_PORT<br>'
            'Gmail → use App Password</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Overlay options ──
    st.markdown("**🎨 Overlay Options**")
    show_heatmap   = st.checkbox("Show Heatmap",         value=False)
    show_tracks    = st.checkbox("Show Tracking Trails", value=True)
    show_crowd_map = st.checkbox("Show Crowd Density",   value=False)
    show_fps_hud   = st.checkbox("Show HUD Overlay",     value=True)
    draw_style     = st.radio("Box Style", ["Corners", "Full Box", "Dot"],
                               label_visibility="collapsed", horizontal=True)

    st.markdown("---")

    # ── Data export ──
    st.markdown("**💾 Export Data**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("📥 CSV", data=db_export_csv(),
                            file_name="deepwatch_log.csv", mime="text/csv",
                            use_container_width=True)
    with col_b:
        st.download_button("📥 JSON", data=db_export_json(),
                            file_name="deepwatch_log.json", mime="application/json",
                            use_container_width=True)

    st.markdown("---")
    if st.button("🗑️ CLEAR SESSION DATA", use_container_width=True):
        for k, v in _defaults.items():
            st.session_state[k] = (v() if callable(v) else
                                   v.copy() if isinstance(v, (dict, deque)) else v)
        st.rerun()


# Global flag — overwritten by sidebar after model selection
IS_WEAPON_MODEL = False

# ─────────────────────────── VIDEO PROCESSOR ──────────────────────────────────
class AdvancedVideoProcessor(VideoProcessorBase):
    """
    Thread-safe video processor with:
    - YOLOv8 inference (every 2nd frame for performance)
    - Simple centroid tracker for person counting and loitering
    - Heatmap accumulation
    - Configurable drawing styles
    - Full HUD overlay
    """

    def __init__(self):
        self._lock       = threading.Lock()
        self.detections  : list  = []
        self.fps         : int   = 0
        self.frame_count : int   = 0
        self.last_frame          = None
        self._t                  = time.time()
        # Centroid tracker: track_id → {"center":..., "age":..., "entry":...}
        self._trackers   : dict  = {}
        self._next_id    : int   = 0
        # Heatmap
        self._heatmap            = None
        self._heatmap_decay      = 0.995
        # Trails: track_id → deque of centers
        self._trails     : dict  = defaultdict(lambda: deque(maxlen=30))
        # Frame dimensions
        self._h = self._w = 0

    # ── centroid tracker ──────────────────────────────────────────────────────
    def _update_tracker(self, new_centers: list[tuple[int,int,str]]) -> dict:
        """Returns map of track_id → (cx,cy,category,entry_time)"""
        MATCH_DIST = 80
        assigned = {}
        free_ids = list(self._trackers.keys())

        for cx, cy, cat in new_centers:
            best_id, best_d = None, MATCH_DIST + 1
            for tid in free_ids:
                tx, ty = self._trackers[tid]["center"]
                d = math.hypot(cx - tx, cy - ty)
                if d < best_d:
                    best_d, best_id = d, tid
            if best_id is not None:
                self._trackers[best_id].update({"center": (cx, cy), "age": 0})
                assigned[best_id] = (cx, cy, cat, self._trackers[best_id]["entry"])
                free_ids.remove(best_id)
            else:
                tid = self._next_id; self._next_id += 1
                self._trackers[tid] = {"center": (cx, cy), "age": 0,
                                       "entry": time.time(), "cat": cat}
                assigned[tid] = (cx, cy, cat, self._trackers[tid]["entry"])

        # Age out missing tracks
        for tid in list(self._trackers.keys()):
            if tid not in assigned:
                self._trackers[tid]["age"] += 1
                if self._trackers[tid]["age"] > 15:
                    del self._trackers[tid]
                    if tid in self._trails:
                        del self._trails[tid]

        return assigned

    # ── draw bounding box ────────────────────────────────────────────────────
    def _draw_box(self, img, x1, y1, x2, y2, color, label, style):
        if style == "Corners":
            blen = min(14, (x2-x1)//3, (y2-y1)//3)
            for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(img,(px,py),(px+dx*blen,py),color,2)
                cv2.line(img,(px,py),(px,py+dy*blen),color,2)
        elif style == "Full Box":
            cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
        else:  # Dot
            cx,cy = (x1+x2)//2,(y1+y2)//2
            cv2.circle(img,(cx,cy),6,color,-1)
            cv2.circle(img,(cx,cy),8,color,1)

        # label background
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)
        cv2.rectangle(img,(x1,y1-th-8),(x1+tw+8,y1),color,-1)
        cv2.putText(img, label, (x1+4,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0), 1, cv2.LINE_AA)

    # ── main recv ────────────────────────────────────────────────────────────
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        now = time.time()
        elapsed = max(now - self._t, 1e-6)
        self.fps = round(1.0 / elapsed)
        self._t  = now

        # Resize for performance
        h, w = img.shape[:2]
        target_w = 640
        if w > target_w:
            scale = target_w / w
            img   = cv2.resize(img, (target_w, int(h * scale)))
        self._h, self._w = img.shape[:2]

        # Init heatmap
        if self._heatmap is None or self._heatmap.shape[:2] != img.shape[:2]:
            self._heatmap = np.zeros((self._h, self._w), dtype=np.float32)

        # ── Inference (every 2nd frame) ───────────────────────────────────
        if self.frame_count % 2 == 0:
            results = model_obj(img, conf=conf_thresh,
                                max_det=max_det, verbose=False)[0]
            new_dets = []
            for box in results.boxes:
                cid   = int(box.cls[0])
                conf  = float(box.conf[0])
                lbl   = model_obj.names[cid]
                cat   = get_category(cid, lbl, IS_WEAPON_MODEL)
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                new_dets.append({
                    "label":    lbl,
                    "conf":     conf,
                    "category": cat,
                    "box":      (x1,y1,x2,y2),
                    "cx":       (x1+x2)//2,
                    "cy":       (y1+y2)//2,
                })
            with self._lock:
                self.detections = new_dets

        with self._lock:
            dets = list(self.detections)

        # ── Heatmap accumulation ──────────────────────────────────────────
        self._heatmap *= self._heatmap_decay
        for d in dets:
            if d["category"] == "person":
                cx, cy = d["cx"], d["cy"]
                r = 30
                y1c = max(0, cy-r); y2c = min(self._h, cy+r)
                x1c = max(0, cx-r); x2c = min(self._w, cx+r)
                self._heatmap[y1c:y2c, x1c:x2c] += 0.15

        # ── Track persons ─────────────────────────────────────────────────
        person_centers = [
            (d["cx"], d["cy"], d["category"])
            for d in dets if d["category"] == "person"
        ]
        track_map = self._update_tracker(person_centers)

        # Update trails
        for tid, (cx,cy,_,_) in track_map.items():
            self._trails[tid].appendleft((cx, cy))

        # ── Draw heatmap ─────────────────────────────────────────────────
        if show_heatmap:
            hm_norm = np.clip(self._heatmap, 0, 1)
            hm_u8   = (hm_norm * 255).astype(np.uint8)
            hm_col  = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
            mask    = hm_u8 > 10
            img[mask] = cv2.addWeighted(img, 0.5, hm_col, 0.5, 0)[mask]

        # ── Draw boundary zones ───────────────────────────────────────────
        zone_breach_labels = []
        for (y_pct, z_label, z_hex) in st.session_state.get("boundary_zones", []):
            ly = int(self._h * (y_pct / 100))
            z_bgr = tuple(int(z_hex.lstrip("#")[i:i+2], 16) for i in (4,2,0))
            # Dashed line
            dash_len, gap_len = 20, 10
            x = 0
            while x < self._w:
                cv2.line(img, (x, ly), (min(x+dash_len, self._w), ly), z_bgr, 2)
                x += dash_len + gap_len
            cv2.putText(img, f"▶ {z_label}", (8, ly - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, z_bgr, 1, cv2.LINE_AA)
            # Check breaches
            for d in dets:
                if d["category"] == "person" and d["cy"] > ly:
                    zone_breach_labels.append(f"{z_label} BREACH")

        # ── Draw trails ───────────────────────────────────────────────────
        if show_tracks:
            for tid, trail in self._trails.items():
                pts = list(trail)
                for i in range(1, len(pts)):
                    alpha = 1.0 - i / len(pts)
                    col = (int(0*alpha), int(180*alpha), int(255*alpha))
                    cv2.line(img, pts[i-1], pts[i], col, 1, cv2.LINE_AA)

        # ── Draw detections ───────────────────────────────────────────────
        loiter_alerts = []
        now_t = time.time()
        for d in dets:
            x1,y1,x2,y2 = d["box"]
            col = CAT_BGR.get(d["category"], (200,200,200))
            label_txt = f"{d['label'].upper()} {int(d['conf']*100)}%"
            self._draw_box(img, x1,y1,x2,y2, col, label_txt, draw_style)

        # Loitering overlay on persons
        for tid, (cx,cy,cat,entry_t) in track_map.items():
            duration = now_t - entry_t
            if cat == "person" and duration > 2:
                dur_txt = f"ID{tid} {int(duration)}s"
                col = (0,0,255) if duration > loiter_thresh else (0,200,255)
                cv2.putText(img, dur_txt, (cx-20, cy+25),
                            cv2.FONT_HERSHEY_SIMPLEX, .38, col, 1, cv2.LINE_AA)
                if duration > loiter_thresh and ac_loiter:
                    loiter_alerts.append(f"LOITERING ID{tid} ({int(duration)}s)")

        # ── Crowd density bar ─────────────────────────────────────────────
        n_people = sum(1 for d in dets if d["category"] == "person")
        if show_crowd_map and n_people > 0:
            bar_h  = 10
            bar_w  = int(min(1.0, n_people / crowd_n) * (self._w - 20))
            bar_col = (0,200,0) if n_people < crowd_n//2 else \
                      (0,150,255) if n_people < crowd_n else (0,0,255)
            cv2.rectangle(img, (10, 10), (10+bar_w, 10+bar_h), bar_col, -1)
            cv2.rectangle(img, (10, 10), (self._w-10, 10+bar_h), (60,60,60), 1)
            cv2.putText(img, f"CROWD: {n_people}/{crowd_n}", (14, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (200,200,200), 1)

        # ── HUD ───────────────────────────────────────────────────────────
        if show_fps_hud:
            hud_h = 36
            ov = img.copy()
            cv2.rectangle(ov, (0, self._h-hud_h), (self._w, self._h), (0,0,0), -1)
            cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)

            hud_txt = (
                f"CAM-01 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"FPS:{self.fps} | OBJ:{len(dets)} | "
                f"{'ACTIVE' if active_now else 'SLEEP'}"
            )
            cv2.putText(img, hud_txt,
                        (8, self._h-10), cv2.FONT_HERSHEY_SIMPLEX, .38,
                        (0,180,255), 1, cv2.LINE_AA)

        # Blinking REC dot
        rec_col = (0,0,200) if int(time.time()*2)%2==0 else (40,40,120)
        cv2.circle(img, (self._w-18, 18), 7, rec_col, -1)
        cv2.putText(img, "REC", (self._w-40,22),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (0,0,200), 1, cv2.LINE_AA)

        # Store frame + extra info
        with self._lock:
            self.last_frame        = img.copy()
            self._zone_breaches    = zone_breach_labels
            self._loiter_alerts    = loiter_alerts
            self._n_people         = n_people

        # Push heatmap snapshot to session state occasionally
        if self.frame_count % 30 == 0:
            hm_u8 = (np.clip(self._heatmap, 0, 1) * 255).astype(np.uint8)
            st.session_state["heatmap"] = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─────────────────────── TOP BAR ─────────────────────────────────────────────
_now_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
_mode_dot = "dot-g" if active_now else "dot-y"
_mode_lbl = "ACTIVE" if active_now else "SCHEDULED SLEEP"

st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">DEEP<em>WATCH</em></div>
    <div class="topbar-badge">v1.0</div>
    <div class="topbar-badge" style="color:#ffc400;border-color:rgba(255,196,0,.35);
         background:rgba(255,196,0,.08)">{op_mode.upper()}</div>
  </div>
  <div class="topbar-status">
    <span><span class="sdot sdot-g"></span>SYS ONLINE</span>
    <span><span class="sdot {_mode_dot}"></span>{_mode_lbl}</span>
    <span><span class="sdot sdot-b"></span>CAM-01 LIVE</span>
    <span style="color:#2a4060">{_now_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not active_now:
    st.markdown(
        '<div class="alert-warn">🌙 System is in SCHEDULED SLEEP mode — '
        'detection is paused until the active window.</div>',
        unsafe_allow_html=True
    )

# ────────────────────── METRIC BAR ───────────────────────────────────────────
uptime   = int(time.time() - st.session_state.start_time)
up_h     = uptime // 3600
up_m     = (uptime % 3600) // 60
up_s     = uptime % 60
up_str   = f"{up_h:02d}:{up_m:02d}:{up_s:02d}"

st.markdown(f"""
<div class="mgrid">
  <div class="mcard" style="--ac:#ff6b6b">
    <div class="mval">{st.session_state.people_count}</div>
    <div class="mlbl">👤 People</div>
    <div class="msub">total this session</div>
  </div>
  <div class="mcard" style="--ac:#00d4ff">
    <div class="mval">{st.session_state.vehicle_count}</div>
    <div class="mlbl">🚗 Vehicles</div>
    <div class="msub">total this session</div>
  </div>
  <div class="mcard" style="--ac:#ff3030">
    <div class="mval">{st.session_state.alert_count}</div>
    <div class="mlbl">🚨 Alerts</div>
    <div class="msub">triggered</div>
  </div>
  <div class="mcard" style="--ac:#ffc400">
    <div class="mval">{st.session_state.snap_count}</div>
    <div class="mlbl">📸 Snapshots</div>
    <div class="msub">auto-captured</div>
  </div>
  <div class="mcard" style="--ac:#00ff88">
    <div class="mval" style="font-size:1.25rem">{up_str}</div>
    <div class="mlbl">⏱ Uptime</div>
    <div class="msub">hh : mm : ss</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────── MAIN LAYOUT ─────────────────────────────────────────
left_col, right_col = st.columns([13, 9], gap="small")

# ── LEFT: VIDEO + SNAPSHOTS ───────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="panel"><div class="panel-hdr">▶ CAM-01 — LIVE FEED'
                '<span class="panel-hdr-badge">RECORDING</span></div>',
                unsafe_allow_html=True)

    facing   = "environment" if "Back" in camera_facing else "user"
    ctx = webrtc_streamer(
        key=f"deepwatch-{facing}-{model_key}",
        video_processor_factory=AdvancedVideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                "facingMode":  {"ideal": facing},
                "width":       {"ideal": 1280},
                "height":      {"ideal": 720},
                "frameRate":   {"ideal": 30},
            },
            "audio": False,
        },
        async_processing=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Snapshots ──────────────────────────────────────────────────────────
    st.markdown('<div class="panel"><div class="panel-hdr">📸 AUTO-CAPTURED SNAPSHOTS'
                f'<span class="panel-hdr-badge">{st.session_state.snap_count} TOTAL</span></div>',
                unsafe_allow_html=True)
    if st.session_state.snapshots:
        snap_list = list(st.session_state.snapshots)[:6]
        n_cols    = min(3, len(snap_list))
        cols      = st.columns(n_cols)
        for idx, snap in enumerate(snap_list):
            with cols[idx % n_cols]:
                st.image(snap["img"],
                         caption=f"{snap['label']} · {snap['time']}",
                         use_container_width=True)
    else:
        st.markdown(
            '<p style="color:#2a4060;font-family:\'Share Tech Mono\',monospace;'
            'font-size:.75rem;padding:4px">No snapshots yet — detection will trigger captures automatically.</p>',
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Zone status ─────────────────────────────────────────────────────────
    if boundary_zones:
        st.markdown('<div class="panel"><div class="panel-hdr">🚧 BOUNDARY ZONE STATUS</div>',
                    unsafe_allow_html=True)
        for y_pct, z_label, z_hex in boundary_zones:
            st.markdown(
                f'<div class="zone-info">▸ {z_label} — boundary at {y_pct}% height</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ── RIGHT: DETECTIONS + LOG + STATS ──────────────────────────────────────────
with right_col:

    # ── Live Detections ─────────────────────────────────────────────────────
    st.markdown('<div class="panel"><div class="panel-hdr">🎯 LIVE DETECTIONS</div>',
                unsafe_allow_html=True)

    alert_placeholder  = st.empty()
    detect_placeholder = st.empty()

    if ctx.video_processor:
        proc  = ctx.video_processor
        with proc._lock:
            dets          = list(proc.detections)
            fps_val       = proc.fps
            last_frm      = proc.last_frame.copy() if proc.last_frame is not None else None
            zone_breaches = getattr(proc, "_zone_breaches", [])
            loiter_alerts = getattr(proc, "_loiter_alerts", [])
            n_people_live = getattr(proc, "_n_people", 0)

        # ── Update session counters ─────────────────────────────────────
        if dets:
            for d in dets:
                st.session_state.label_counts[d["label"]] += 1
                if d["category"] == "person":
                    st.session_state.people_count += 1
                elif d["category"] == "vehicle":
                    st.session_state.vehicle_count += 1
            st.session_state.total_detections += len(dets)

        # ── Crowd alert ─────────────────────────────────────────────────
        crowd_alert = ""
        if ac_crowd and n_people_live >= crowd_n:
            crowd_alert = f"CROWD DENSITY ALERT — {n_people_live} PEOPLE IN FRAME"

        # ── WEAPON DETECTION — Instant priority alert ────────────────────
        weapon_dets = [d for d in dets if d["category"] == "weapon"]
        now_t2 = time.time()

        if weapon_dets and last_frm is not None:
            # Always show weapon alert — separate cooldown (10s)
            if now_t2 - st.session_state.get("last_weapon_alert", 0) > 10:
                st.session_state["last_weapon_alert"] = now_t2
                st.session_state.alert_count += 1

                weapon_labels = ", ".join(
                    f"{d['label'].upper()} ({int(d['conf']*100)}%)"
                    for d in weapon_dets
                )
                weapon_html = (
                    f'<div style="background:linear-gradient(135deg,rgba(255,0,0,.35),'
                    f'rgba(180,0,0,.25));border:2px solid #ff0000;border-radius:8px;'
                    f'padding:12px 16px;margin:5px 0;font-family:\'Share Tech Mono\','
                    f'monospace;font-size:.85rem;color:#ff3030;'
                    f'animation:apulse .5s ease infinite alternate;">'
                    f'🔫 WEAPON DETECTED — {weapon_labels}</div>'
                )
                alert_placeholder.markdown(weapon_html, unsafe_allow_html=True)

                # Log weapon event
                st.session_state.event_log.appendleft({
                    "time":  datetime.datetime.now().strftime("%H:%M:%S"),
                    "msg":   f"🔫 WEAPON — {weapon_labels}",
                    "level": "critical",
                    "count": len(weapon_dets),
                })
                threading.Thread(
                    target=db_insert,
                    args=("critical", "weapon", weapon_labels,
                          f"WEAPON DETECTED: {weapon_labels}",
                          max(d["conf"] for d in weapon_dets), "CAM-01"),
                    daemon=True
                ).start()

                # ── Weapon Snapshot (always, no cooldown check) ──────────
                _, buf      = cv2.imencode(".jpg", last_frm, [cv2.IMWRITE_JPEG_QUALITY, 95])
                snap_bytes  = buf.tobytes()
                snap_rgb    = cv2.cvtColor(last_frm, cv2.COLOR_BGR2RGB)
                st.session_state.snapshots.appendleft({
                    "img":   snap_rgb,
                    "time":  datetime.datetime.now().strftime("%H:%M:%S"),
                    "label": f"🔫 {weapon_labels}",
                    "bytes": snap_bytes,
                })
                st.session_state.snap_count += 1
                st.session_state.last_snap_time = now_t2

                # ── Weapon Email (instant, always sends) ─────────────────
                if email_enabled:
                    max_conf_w = max(d["conf"] for d in weapon_dets)
                    _wsubj = f"🔫 WEAPON DETECTED on CAM-01 — IMMEDIATE ACTION REQUIRED"
                    _wbody = (
                        f"⚠️  WEAPON ALERT — DeepWatch Security System\n"
                        f"{'='*50}\n\n"
                        f"🔫 Weapon(s) Detected: {weapon_labels}\n"
                        f"📊 Confidence: {max_conf_w:.0%}\n"
                        f"📷 Camera: CAM-01\n"
                        f"🕐 Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        f"{'='*50}\n"
                        f"Snapshot attached. Immediate action recommended.\n"
                        f"DeepWatch v1.0 — Automated Security Alert\n"
                    )
                    def _send_weapon(subj, body, img_b):
                        ok, msg = send_email_alert(subj, body, img_b)
                        st.session_state.email_count += 1
                        st.session_state.event_log.appendleft({
                            "time":  datetime.datetime.now().strftime("%H:%M:%S"),
                            "msg":   f"🔫 Weapon email {'✓ sent' if ok else '✗ FAILED: '+msg}",
                            "level": "info" if ok else "critical",
                            "count": 0,
                        })
                    threading.Thread(
                        target=_send_weapon,
                        args=(_wsubj, _wbody, snap_bytes),
                        daemon=True
                    ).start()

        # ── Trigger regular alerts ───────────────────────────────────────
        hit_cats     = {d["category"] for d in dets} & st.session_state.alert_cats
        all_alerts   = []
        if hit_cats and "weapon" not in hit_cats:  # weapon handled above
            all_alerts += [f"DETECTED — {', '.join(sorted(hit_cats)).upper()}"]
        if zone_breaches:
            all_alerts += zone_breaches
        if loiter_alerts:
            all_alerts += loiter_alerts
        if crowd_alert:
            all_alerts.append(crowd_alert)

        if all_alerts and (now_t2 - st.session_state.last_alert_time > alert_cooldown):
            st.session_state.last_alert_time = now_t2
            st.session_state.alert_count    += 1

            alert_html = "".join(
                f'<div class="alert-crit">🚨 {a}</div>' for a in all_alerts
            )
            if not weapon_dets:  # don't overwrite weapon alert
                alert_placeholder.markdown(alert_html, unsafe_allow_html=True)

            # Log to deque + DB
            for a in all_alerts:
                entry = {
                    "time":  datetime.datetime.now().strftime("%H:%M:%S"),
                    "msg":   a,
                    "level": "critical",
                    "count": len(dets),
                }
                st.session_state.event_log.appendleft(entry)
                threading.Thread(
                    target=db_insert,
                    args=("critical", "alert", "ALERT", a, 1.0, "CAM-01"),
                    daemon=True
                ).start()

            # ── Snapshot for non-weapon alerts ────────────────────────
            snap_hit = {d["category"] for d in dets} & st.session_state.snap_cats
            snap_hit.discard("weapon")  # weapon snapshot already done above
            if snap_hit and last_frm is not None:
                if now_t2 - st.session_state.last_snap_time > snap_cooldown:
                    st.session_state.last_snap_time = now_t2
                    st.session_state.snap_count    += 1

                    _, buf      = cv2.imencode(".jpg", last_frm)
                    snap_bytes  = buf.tobytes()
                    snap_rgb    = cv2.cvtColor(last_frm, cv2.COLOR_BGR2RGB)
                    hit_labels  = ", ".join(
                        d["label"].upper() for d in dets
                        if d["category"] in snap_hit
                    )
                    st.session_state.snapshots.appendleft({
                        "img":   snap_rgb,
                        "time":  datetime.datetime.now().strftime("%H:%M:%S"),
                        "label": hit_labels,
                        "bytes": snap_bytes,
                    })

                    # ── Email ─────────────────────────────────────────
                    if email_enabled:
                        max_conf = max(
                            (d["conf"] for d in dets if d["category"] in snap_hit),
                            default=0
                        )
                        email_subj = f"{hit_labels} detected on CAM-01"
                        email_body = (
                            f"Detection: {hit_labels}\n"
                            f"Confidence: {max_conf:.0%}\n"
                            f"Zone breaches: {', '.join(zone_breaches) or 'None'}\n"
                            f"Loitering: {', '.join(loiter_alerts) or 'None'}\n"
                            f"Time: {datetime.datetime.now().isoformat()}\n"
                            f"Camera: CAM-01\n"
                        )
                        def _send(subj, body, img_b):
                            ok, msg = send_email_alert(subj, body, img_b)
                            st.session_state.email_count += 1
                            lv = "info" if ok else "critical"
                            st.session_state.event_log.appendleft({
                                "time":  datetime.datetime.now().strftime("%H:%M:%S"),
                                "msg":   f"Email {'✓ sent' if ok else '✗ FAILED: '+msg}",
                                "level": lv,
                                "count": 0,
                            })
                        threading.Thread(
                            target=_send,
                            args=(email_subj, email_body, snap_bytes),
                            daemon=True
                        ).start()

        # ── Object tags + confidence bars ────────────────────────────
        if dets:
            tags_html = "".join(
                f'<span class="otag t-{d["category"]}">'
                f'{CAT_ICON.get(d["category"],"📦")} {d["label"]} '
                f'{int(d["conf"]*100)}%</span>'
                for d in sorted(dets, key=lambda x: -x["conf"])
            )

            bars_html = ""
            for d in sorted(dets, key=lambda x: -x["conf"])[:10]:
                pct = int(d["conf"] * 100)
                hx  = CAT_HEX.get(d["category"], "#00c8ff")
                bars_html += f"""
                <div style="margin:4px 0">
                  <div style="display:flex;justify-content:space-between;
                              font-family:'Share Tech Mono',monospace;font-size:.69rem;color:#5a7090">
                    <span>{d['label'].upper()}</span>
                    <span style="color:{hx}">{pct}%</span>
                  </div>
                  <div class="cbar-wrap">
                    <div class="cbar" style="width:{pct}%;background:linear-gradient(90deg,{hx},{hx}88)"></div>
                  </div>
                </div>"""

            fps_html = (
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.65rem;'
                f'color:#2a4060;margin-top:6px">FPS: {fps_val} | '
                f'FRAME: {proc.frame_count} | OBJECTS: {len(dets)}</div>'
            )
            detect_placeholder.markdown(
                tags_html + "<br>" + bars_html + fps_html, unsafe_allow_html=True
            )
        else:
            detect_placeholder.markdown(
                '<p style="color:#1e3050;font-family:\'Share Tech Mono\',monospace;'
                'font-size:.75rem;padding:4px">NO OBJECTS IN FRAME</p>',
                unsafe_allow_html=True
            )
    else:
        detect_placeholder.markdown(
            '<p style="color:#1e3050;font-family:\'Share Tech Mono\',monospace;'
            'font-size:.75rem;padding:4px">▶ PRESS START TO ACTIVATE FEED</p>',
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Session Statistics ───────────────────────────────────────────────────
    st.markdown('<div class="panel"><div class="panel-hdr">📊 SESSION STATISTICS'
                f'<span class="panel-hdr-badge">{st.session_state.total_detections} TOTAL</span></div>',
                unsafe_allow_html=True)

    if st.session_state.label_counts:
        top_items = sorted(
            st.session_state.label_counts.items(), key=lambda x: -x[1]
        )[:10]
        max_c = max(c for _, c in top_items)

        bars = ""
        for label, count in top_items:
            cid_match = next(
                (i for i, n in model_obj.names.items() if n == label), 999
            )
            cat = get_category(cid_match)
            hx  = CAT_HEX.get(cat, "#00c8ff")
            bw  = int(count / max_c * 100)
            bars += f"""
            <div style="margin:5px 0">
              <div style="display:flex;justify-content:space-between;
                          font-family:'Share Tech Mono',monospace;font-size:.69rem;color:#5a7090">
                <span>{CAT_ICON.get(cat,'📦')} {label.upper()}</span>
                <span style="color:{hx};font-weight:700">{count:,}×</span>
              </div>
              <div class="cbar-wrap">
                <div class="cbar" style="width:{bw}%;background:linear-gradient(90deg,{hx},{hx}44)"></div>
              </div>
            </div>"""
        st.markdown(bars, unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="color:#1e3050;font-family:\'Share Tech Mono\',monospace;'
            'font-size:.75rem">Awaiting detections…</p>',
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Event Log ───────────────────────────────────────────────────────────
    st.markdown('<div class="panel"><div class="panel-hdr">📋 EVENT LOG'
                '<span class="panel-hdr-badge">LIVE</span></div>',
                unsafe_allow_html=True)

    log_tab1, log_tab2 = st.tabs(["Live (Session)", "Persisted (DB)"])

    with log_tab1:
        if st.session_state.event_log:
            lvl_map = {"critical":"lc","warning":"lw","info":"li","success":"ls"}
            html = ""
            for e in list(st.session_state.event_log)[:20]:
                cls = lvl_map.get(e.get("level","info"), "li")
                cnt = f"+{e['count']}" if e.get("count") else ""
                msg = e["msg"][:60] + ("…" if len(e["msg"])>60 else "")
                html += f"""
                <div class="log-row {cls}">
                  <span class="log-t">{e['time']}</span>
                  <span class="log-m">{msg}</span>
                  <span class="log-n">{cnt}</span>
                </div>"""
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color:#1e3050;font-family:\'Share Tech Mono\',monospace;'
                'font-size:.75rem">Log is empty</p>',
                unsafe_allow_html=True
            )

    with log_tab2:
        db_rows = db_fetch_recent(30)
        if db_rows:
            lvl_map = {"critical":"lc","warning":"lw","info":"li"}
            html = ""
            for row in db_rows:
                ts, level, cat, label, msg, conf, cam = row
                cls = lvl_map.get(level, "li")
                t_short = ts[11:19]
                conf_s  = f"{conf:.0%}" if conf else ""
                html += f"""
                <div class="log-row {cls}">
                  <span class="log-t">{t_short}</span>
                  <span class="log-m">[{cam}] {msg[:45]}</span>
                  <span class="log-n" style="color:#5a7090">{conf_s}</span>
                </div>"""
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color:#1e3050;font-family:\'Share Tech Mono\',monospace;'
                'font-size:.75rem">No persisted events yet</p>',
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────── FOOTER ──────────────────────────────────────────
st.markdown("""
<div class="vfooter">
  DEEPWATCH v1.0  ·  YOLOv8  ·  STREAMLIT  ·  OPENCV  ·  WEBRTC  ·  SQLITE
  &nbsp;|&nbsp; EU AI ACT COMPLIANT PORTFOLIO PROJECT and This is just for Educational Purposses ·  2026.created by Raja Roy
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# streamlit run app_version2.py