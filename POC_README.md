# Dog detection + nervousness POC

Quick CPU-friendly POC that connects to a camera, detects moving objects (candidate dogs), tracks them, and computes a simple "nervousness" score based on movement.

Setup

1. Create a Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

2. Edit `config/camera.yml` and set `source` to `0` (webcam) or your camera RTSP/HTTP URL.

Run

```powershell
python src\run_poc.py --config config\camera.yml
```

Notes

- By default the POC uses motion detection as a proxy to find moving animals. For better accuracy provide an ONNX object detector and extend `detector.py`.
- Audio support is not implemented in this initial POC but hooks are left in config.
- The nervousness heuristic is simple (movement speed). This is intended as a fast-to-run baseline.
