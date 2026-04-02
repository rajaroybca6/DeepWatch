# DeepWatch v1.0 — Full Setup Guide

```
██████╗ ███████╗███████╗██████╗ ██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
██║  ██║█████╗  █████╗  ██████╔╝██║ █╗ ██║███████║   ██║   ██║     ███████║
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
██████╔╝███████╗███████╗██║     ╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
╚═════╝ ╚══════╝╚══════╝╚═╝      ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
```

**Deep Learning Meets Real-Time Security**

**Stack:** Python 3.10+ · Streamlit · YOLOv8 · OpenCV · WebRTC · SQLite

---

## 📁 Project Structure

```
your-project/
│
├── app.py                      ← Main application
├── requirements.txt            ← Python dependencies
├── deepwatch_events.db          ← Auto-created SQLite database (after first run)
│
└── .streamlit/
    ├── config.toml             ← Streamlit server/theme config
    └── secrets.toml            ← SMTP + TURN credentials (never commit this!)
```

---

## ⚡ Quick Start (Local)

### 1. Clone / place your files
```bash
mkdir deepwatch && cd deepwatch
# copy app.py, requirements.txt, and .streamlit/ folder here
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** On some systems you may need to use `pip3` instead of `pip`.
> If you have a GPU and want faster inference, install the GPU build of PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Configure secrets
```bash
# Copy the template and fill in your values:
cp .streamlit/secrets.toml .streamlit/secrets.toml
# Then edit .streamlit/secrets.toml in any text editor
```

### 5. Run the app
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

1. Push your project to a **GitHub repository**
   - ⚠️ Add `.streamlit/secrets.toml` to `.gitignore` — never push credentials!

2. Go to **https://share.streamlit.io** → New app → select your repo

3. In **Advanced settings → Secrets**, paste the contents of your `secrets.toml`

4. Click **Deploy** — done!

---

## 🔑 Gmail App Password (for Email Alerts)

1. Go to your Google Account → **Security**
2. Enable **2-Step Verification** (required)
3. Search for **"App passwords"**
4. Generate a new password for "Mail" / "Other (custom name)"
5. Copy the 16-character password into `SMTP_PASS` in `secrets.toml`

---

## 📡 TURN Server (for WebRTC behind firewalls)

If the camera feed doesn't start (common on corporate networks, some mobile carriers):

1. Sign up free at **https://dashboard.metered.ca**
2. Go to **Realtime → TURN Credentials**
3. Copy the TURN URLs, username, and password into `secrets.toml`
4. Set `FORCE_TURN = "true"` if still having issues

The app falls back to free OpenRelay TURN servers automatically if no custom TURN is configured.

---

## 🤖 YOLO Model Weights

Models are downloaded automatically on first run from Ultralytics:

| Model | Speed | Accuracy | File size |
|-------|-------|----------|-----------|
| `yolov8n.pt` | ⚡⚡⚡ Fastest | ★★☆ | ~6 MB  |
| `yolov8s.pt` | ⚡⚡  Balanced | ★★★ | ~22 MB |
| `yolov8m.pt` | ⚡   Accurate | ★★★★| ~50 MB |

Switch between them live in the sidebar — they are cached after first load.

To use a **custom-trained model** (e.g., for weapons, hard hats, fire):
```python
# In app.py, change MODEL_OPTIONS:
MODEL_OPTIONS = {
    "My Custom Model": "path/to/best.pt",
    ...
}
```

---

## ⚙️ Feature Reference

| Feature | Where to configure |
|---|---|
| Camera (front/rear) | Sidebar → Camera Feed |
| Model quality | Sidebar → Model |
| Detection confidence | Sidebar → Detection Confidence slider |
| Operation mode (schedule) | Sidebar → Operation Mode |
| Boundary zones (up to 4) | Sidebar → Boundary Zones |
| Alert categories | Sidebar → Alert Triggers |
| Alert email cooldown | Sidebar → Alert cooldown |
| Loitering threshold | Sidebar → Loitering Detection |
| Auto-snapshot categories | Sidebar → Auto Snapshot |
| Email alerts | Sidebar → Email Alerts (+ secrets.toml) |
| Heatmap / trails / HUD | Sidebar → Overlay Options |
| Box drawing style | Sidebar → Box Style |
| Export logs (CSV/JSON) | Sidebar → Export Data |
| Clear session data | Sidebar → CLEAR SESSION DATA button |

---

## 🗄️ Database

Events are persisted to `deepwatch_events.db` (SQLite, auto-created):

```sql
SELECT * FROM events ORDER BY id DESC LIMIT 50;
```

Schema:
```
id        INTEGER  PRIMARY KEY
ts        TEXT     ISO-8601 timestamp
level     TEXT     critical / warning / info
category  TEXT     person / vehicle / weapon / object / alert
label     TEXT     YOLO class name
message   TEXT     human-readable event description
conf      REAL     detection confidence (0–1)
cam       TEXT     camera identifier (default: CAM-01)
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| Camera feed doesn't start | Allow camera permission in browser; try a different browser (Chrome works best) |
| Black screen / frozen feed | Reload page; check browser console for WebRTC errors |
| Feed works locally but not on cloud | Add TURN server credentials to secrets.toml |
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python-headless` |
| `ModuleNotFoundError: av` | Run `pip install av` |
| Email not sending | Double-check App Password; ensure 2FA is ON for Gmail |
| YOLO model download fails | Check internet connection; models auto-download to `~/.ultralytics/` |
| App crashes on Apple Silicon | Use `pip install opencv-python` instead of headless version |

---

## 📋 EU AI Act Notes (Portfolio)

This project is for **portfolio / educational purposes**.
For real-world deployment:
- Obtain explicit consent before recording individuals
- Display visible CCTV notices (GDPR Article 13)
- Apply data minimisation — don't store raw footage unnecessarily
- Conduct a DPIA (Data Protection Impact Assessment) for biometric data
- Comply with local laws on facial recognition and automated surveillance

---

*DeepWatch v1.0 · YOLOv8 · Streamlit · OpenCV · WebRTC · SQLite*