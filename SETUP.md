# Smart Vision System – Setup Guide (Windows 11)

## Prerequisites
- Python 3.10 or 3.11 (recommended)
- CUDA 12.x (for RTX 5070 Ti)
- Git (optional)

---

## Step 1 – Install Python
Download from https://www.python.org/downloads/
✅ Check "Add Python to PATH" during installation.

---

## Step 2 – Create a Virtual Environment
Open Command Prompt or PowerShell in the project folder:

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Step 3 – Install PyTorch with CUDA (RTX 5070 Ti)
RTX 5070 Ti uses CUDA 12.x. Run:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Step 4 – Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5 – Verify GPU is Detected

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 5070 Ti
```

---

## Step 6 – Run the Project

### Option A – Interactive Menu (Recommended for beginners)
```bash
python main.py
```
Then follow the on-screen menu to select mode and input source.

### Option B – Direct CLI Launch
```bash
# Traffic mode with video file
python main.py --mode traffic --source "C:\path\to\video.mp4"

# Surveillance mode with webcam
python main.py --mode surveillance --source 0

# Save output video
python main.py --mode traffic --source "video.mp4" --save

# Custom confidence threshold
python main.py --mode surveillance --source 0 --conf 0.5
```

---

## Keyboard Controls (while video is running)

| Key | Action |
|-----|--------|
| Q   | Quit / stop processing |
| S   | Save snapshot to output/ |
| H   | Toggle heatmap (Surveillance only) |
| R   | Reset vehicle counters (Traffic only) |

---

## Project Structure

```
smart_vision_system/
├── main.py                    ← Entry point
├── requirements.txt
├── config/
│   └── settings.py            ← All tunable parameters
├── core/
│   ├── detector.py            ← YOLOv8 wrapper
│   ├── tracker.py             ← DeepSORT tracker
│   ├── reid.py                ← Re-identification
│   └── video_io.py            ← Camera / file I/O
├── traffic/
│   └── traffic_system.py      ← Traffic management pipeline
├── surveillance/
│   └── surveillance_system.py ← Surveillance pipeline
├── utils/
│   ├── display.py             ← CLI colours & formatting
│   └── logger.py              ← File + console logging
├── models/
│   └── weights/               ← Auto-downloaded YOLO weights
└── output/
    ├── recordings/            ← Saved output videos
    └── logs/                  ← Session log files
```

---

## Tuning Parameters
Edit `config/settings.py` to adjust:
- `model_size` – yolov8n (fastest) to yolov8x (most accurate)
- `confidence_threshold` – lower = more detections, higher = fewer false positives
- `speed_limit_kmh` – adjust for your use case
- `loitering_time_threshold` – seconds before loitering alert fires
- `fall_aspect_ratio_threshold` – tune for camera angle

---

## Troubleshooting

**"CUDA out of memory"**
→ Switch to a smaller model: set `model_size = "yolov8s"` in settings.py

**"Cannot open video source"**
→ Check the file path. Use full absolute path with forward slashes or double backslashes.

**Low FPS**
→ Reduce `input_size` from 640 to 480 or 320 in settings.py

**Webcam not detected**
→ Try source `1` or `2` instead of `0`
