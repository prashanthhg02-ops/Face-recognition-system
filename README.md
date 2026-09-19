# Face-recognition-system
# Face Recognition System

A local, consent-based face recognition demo built with Python, OpenCV, and machine-learning-based LBPH recognition. It detects faces with OpenCV's Haar cascade, learns from labeled camera samples, and identifies known people locally.

## Setup

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `.venv\Scripts\python.exe` directly instead.

## Run

1. Capture samples for each person, with their permission:

```powershell
python app.py register --name Alice
```

2. Train the model:

```powershell
python app.py train
```

3. Start recognition:

```powershell
python app.py recognize
```

Press `q` in an OpenCV window to stop. Use `--camera 1` for a second camera. The `--threshold` option controls strictness; lower values are stricter and reduce false matches.

## Project data

Captured images, the trained model, and label metadata are stored in `data/`, which is ignored by Git. Do not collect or share face data without informed consent, and do not use this demo as the sole basis for access, employment, law-enforcement, or other high-impact decisions.

## Troubleshooting

- `Could not open camera`: check camera permissions, close other camera applications, or try `--camera 1`.
- `module 'cv2' has no attribute 'face'`: uninstall `opencv-python` if it is installed alongside `opencv-contrib-python`, then reinstall the requirements.
- Recognition is inaccurate: capture samples in the same lighting and camera position, add more samples, and lower the threshold.
