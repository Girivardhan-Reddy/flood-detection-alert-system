# 🌊 Flood Detection Alert System

An AI-powered Flood Detection and Alert System built using **Python, Flask, Vision Transformers (ViT), OpenCV, and Weather APIs**.

This system detects flood conditions from uploaded images/videos, analyzes environmental risk using real-time weather data, generates Grad-CAM visualizations, creates PDF reports, and provides location-based flood alerts.

---

# 🚀 Features

## ✅ AI Flood Detection
- Detects Flood / Non-Flood scenes
- Uses Vision Transformer (ViT) deep learning model
- Real-time prediction probabilities

## ✅ Grad-CAM Visualization
- Attention heatmaps for explainable AI
- Highlights flood-affected regions in images

## ✅ Weather Integration
- Real-time weather analysis using OpenWeather API
- Rainfall, humidity, pressure, wind speed monitoring
- Flood risk assessment

## ✅ Location Detection
- IP-based location detection
- Browser geolocation support
- EXIF GPS extraction from uploaded images
- Manual city/location search

## ✅ Video Processing
- Frame-by-frame flood detection
- Flood percentage analysis
- Key flood frame extraction

## ✅ Interactive Maps
- Flood location mapping using Folium
- Risk-level colored markers

## ✅ PDF Report Generation
- Detailed flood analysis reports
- Weather and risk assessment
- Emergency recommendations

## ✅ Database Support
- SQLite database integration
- Detection history storage
- Location history tracking

---

# 🛠 Tech Stack

## Backend
- Python
- Flask

## AI / Deep Learning
- PyTorch
- timm (Vision Transformer)
- OpenCV
- torchvision

## Database
- SQLite

## Visualization
- Matplotlib
- Seaborn
- Grad-CAM

## APIs
- OpenWeather API
- IP Geolocation API

## Other Libraries
- Folium
- Geopy
- ReportLab
- PIL

---

# 📂 Project Structure

```bash
flood-detection-alert-system/
│
├── static/
│   ├── uploads/
│   ├── processed/
│   └── reports/
│
├── templates/
│   ├── index.html
│   ├── view_detection.html
│   └── 404.html
│
├── flood_vit_model.pth
├── flood_detection.db
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Girivardhan-Reddy/flood-detection-alert-system.git
```

## 2️⃣ Move into Project

```bash
cd flood-detection-alert-system
```

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

## 4️⃣ Activate Environment

### Windows
```bash
venv\Scripts\activate
```

### Linux / Mac
```bash
source venv/bin/activate
```

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python app.py
```

Open browser:

```bash
http://localhost:5000
```

---

# 🔑 OpenWeather API Setup

Replace this line in `app.py`:

```python
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY_HERE"
```

with your API key from:

https://openweathermap.org/api

---

# 🧠 AI Model

The project uses:

- Vision Transformer Tiny Patch16 224
- Binary Classification:
  - Flood
  - NonFlood

---

# 📊 Flood Risk Levels

| Flood Probability | Severity |
|------------------|----------|
| 0–30% | SAFE |
| 30–50% | LOW |
| 50–70% | MODERATE |
| 70–85% | HIGH |
| 85%+ | EXTREME |

---

# 📸 Supported File Types

## Images
- JPG
- JPEG
- PNG
- BMP
- GIF

## Videos
- MP4
- AVI
- MOV
- MKV
- WEBM

---

# 📄 Generated Reports

The system automatically generates:
- Flood analysis reports
- Weather reports
- Risk assessments
- Emergency recommendations

PDF reports are stored in:

```bash
static/reports/
```

---

# 🗺 Location Features

- Automatic IP tracking
- Browser GPS support
- City search
- EXIF metadata extraction
- Interactive flood maps

---

# 🔥 Future Improvements

- SMS alert integration
- Email notification system
- Live CCTV flood monitoring
- Drone flood detection
- IoT sensor integration
- Mobile application

---

# 👨‍💻 Author

## Girivardhan Reddy

AI & Full Stack Developer  
Machine Learning & Computer Vision Enthusiast

GitHub:
https://github.com/Girivardhan-Reddy

---

# ⭐ If You Like This Project

Give this repository a ⭐ on GitHub.

---

# 📜 License

This project is licensed under the MIT License.
