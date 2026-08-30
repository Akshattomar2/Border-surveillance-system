<div align="center">

# 🛡️ IBVAP
### Intelligent Border Video Analytics Platform

**Turning ordinary CCTV feeds into an AI-powered border surveillance system — no specialized hardware required.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#-license)
[![SIH 2026](https://img.shields.io/badge/Smart%20India%20Hackathon-2026-orange)](#)

<img src="https://img.shields.io/badge/status-active%20development-brightgreen" alt="status"/>

</div>

---

## 📌 Overview

Border security forces rely on CCTV, but conventional systems only **record** — they don't **think**.
Advanced features like facial recognition, ANPR, and intrusion detection usually need expensive, proprietary hardware — hard to deploy at scale in remote border areas.

**IBVAP** is a software-only AI layer that plugs into *existing* IP cameras and turns them into an intelligent surveillance network — detecting people, vehicles, and fence intrusions in real time, with zero extra hardware.

---

## ✨ Features

| Capability | Status |
|---|:---:|
| 👤 Human & vehicle detection + tracking | ✅ |
| 🚧 Virtual fence / intrusion detection | ✅ |
| 🧗 Custom-trained fence-climbing detection (YOLOv8) | ✅ |
| 🙂 Face detection (identity recognition not enabled) | ✅ |
| ✅ Authorized-person enrollment and face matching | ✅ |
| ⚠️ Suspicious activity / loitering heuristic | ✅ |
| 🌙 Night-time movement detection | ✅ |
| 📊 Real-time alert logging (CSV + snapshots) | ✅ |
| 📹 Multi-video upload and sequential analysis | ✅ |
| 🚗 Automatic Number Plate Recognition (ANPR) | ✅ |
| 🧍 Advanced face recognition / liveness controls | ✅ |

---

## 🎬 Demo

<div align="center">
<i>Upload a video, hit play, and watch detections happen live —<br/>scrub the timeline to re-analyze any moment.</i>

<br/><br/>

`🟢 Person Detected`&nbsp;&nbsp;&nbsp;`🔴 Fence Intrusion Alert`&nbsp;&nbsp;&nbsp;`🟠 Climbing Behavior Flagged`

</div>

---

## 🏗️ Architecture

```
                    ┌─────────────────┐
   CCTV / Video ───▶│  Frame Ingestion │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │   YOLOv8 Model   │  (custom-trained on
                    │   Inference      │   fence/intrusion data)
                    └────────┬────────┘
                             ▼
              ┌──────────────┴───────────────┐
              ▼                               ▼
     ┌─────────────────┐           ┌────────────────────┐
     │ Virtual Fence /  │           │  Alert Engine       │
     │ Zone Logic       │──────────▶│  (CSV log + snapshot)│
     └─────────────────┘           └──────────┬──────────┘
                                               ▼
                                    ┌─────────────────────┐
                                    │ Streamlit Dashboard  │
                                    │ (play/pause/scrub)   │
                                    └─────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/Akshattomar2/Border-surveillance-system.git
cd Border-surveillance-system

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Add a test video

Drop any CCTV-style or street footage into the project folder as `test.mp4`
(or just upload one directly from the dashboard sidebar).

### 3. (Optional) Train the custom fence-detection model

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 name=fence_model
```

The dashboard automatically detects and uses your trained weights
(`runs/detect/<run_name>/weights/best.pt`) if available — otherwise it
falls back to pretrained YOLOv8.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open the browser tab → hit **▶️ Play** → watch real-time detection,
or drag the **timeline slider** to jump to any frame and re-analyze it instantly.

---

## 🎮 Using the Dashboard

| Control | What it does |
|---|---|
| ▶️ **Play** | Streams the video with live AI detection overlaid |
| ⏸️ **Pause** | Freezes on the current frame |
| ⏮️ **Restart** | Jumps back to frame 0 |
| 🎚️ **Timeline Slider** | Scrub to any point and re-analyze that exact frame — as many times as you want |
| 📤 **Upload Video** | Swap in a new video without touching any code |
| 🗑️ **Clear Alerts** | Resets the alert log |

---

## 🧠 Tech Stack

- **Computer Vision:** YOLOv8 (Ultralytics), OpenCV
- **Dashboard:** Streamlit
- **Data:** Roboflow-annotated fence/intrusion datasets
- **Language:** Python 3.12

---

## 📁 Project Structure

```
Border-surveillance-system/
├── detector.py          # Core detection logic (YOLO + fence/zone alerts)
├── app.py                # Interactive Streamlit dashboard
├── requirements.txt      # Python dependencies
├── data.yaml              # Dataset config for training
├── train/ valid/ test/    # Dataset splits (images + YOLO labels)
├── runs/detect/           # Training outputs (auto-generated)
├── alerts.csv             # Logged intrusion events (auto-generated)
└── snapshots/              # Saved alert snapshots (auto-generated)
```

---

## 🗺️ Roadmap

- [ ] ANPR (Automatic Number Plate Recognition)
- [ ] Face detection & recognition layer
- [ ] Night-vision / low-light detection mode
- [ ] Object tracking (ByteTrack) to reduce duplicate alerts
- [ ] Live RTSP camera feed support
- [ ] Multi-camera dashboard view

---

## 🤝 Contributing

This project is being built for **Smart India Hackathon 2026**.
Pull requests and suggestions are welcome — open an issue or submit a PR.

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">



</div>
