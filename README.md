# FaceFuck

Write Brainfuck with your face. Yes, your face.

FaceFuck is a PyQt5 “IDE” that turns **facial expressions** and **head position** into Brainfuck commands using **DeepFace** + **OpenCV**.

If you came here for productivity: please reconsider.\
If you came here for vibes: welcome home.

## What You Get

- Webcam → DeepFace emotion analysis → Brainfuck input (hands optional)
- Noise resistance (smoothing + hysteresis) to reduce “emotion whiplash”
- Multi-face handling (sticks to one face instead of playing musical chairs)
- Head-position commands (`[` / `]`) + up/down control (Run/Stop)
- On-screen 3×3 grid overlay (aiming assistance for left/center/right + up/middle/down)
- Live tape view (Brainfuck memory grid) and runtime error reporting

## Requirements

- OS: Windows recommended (works best with DirectShow). Linux/macOS may work with minor tweaks.
- Python: 3.10+ recommended
- Webcam access permission
- Dependencies (see `requirements.txt`):
  - `opencv-python`
  - `deepface`
  - `pyqt5`

DeepFace will download models on first run unless you point it to a local cache.

## Install

### Option A: venv (recommended)

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

### Option B: conda

```powershell
conda create -n facefuck python=3.10 -y
conda activate facefuck
pip install -r requirements.txt
```

### DeepFace model cache (recommended)

To keep models inside the repo (or any folder you like), set:

```powershell
$env:DEEPFACE_HOME = (Resolve-Path .).Path
```

`main.py` already sets `DEEPFACE_HOME` to the project directory by default.

## Run

```powershell
python main.py
```

## How To Use (The “No Keyboard” Workflow)

1. Put your face in the camera view.
2. Keep your face **in the middle third** (vertical middle) to enable code input.
3. Hold a stable trigger for \~**1.5s** to commit a Brainfuck symbol.
4. Move your face **up** and hold briefly (\~**0.8s**) to **RUN**.
5. Move your face **down** and hold briefly (\~**0.8s**) to **STOP**.

The UI shows:

- A face bounding box with `(position / emotion)` label
- A single HOLD progress bar (sky-blue during up/down control)
- A tape grid (memory cells) as the main runtime visualization

## Mapping (Current)

### Position (priority)

- **left** → `[`
- **right** → `]`

### Emotion (only when position is center)

- **happy** → `+` (INC)
- **sad** → `-` (DEC)
- **angry** → `<` (PTR\_LEFT)
- **surprise** → `>` (PTR\_RIGHT)
- **fear** → `,` (IN)
- **disgust** → `.` (OUT)
- **neutral** → `BACKSPACE` (delete last symbol)

## Example Assets (Reference Images)

The left “Sample” column loads images from:

- `icon/<key>.jpg|png`

Keys include:

- `left`, `right`, `up`, `down`, `center` (or `middle`)
- `happy`, `sad`, `angry`, `surprise`, `fear`, `disgust`, `neutral`

Missing images are shown as empty cells (no drama).

## Tuning (If Your Face Is Too Powerful)

If input feels too jumpy or too slow, tweak these in `src/qt_ui.py`:

- `analysis_interval` (how often DeepFace runs)
- `hold_duration` (symbol commit time)
- `vertical_hold_duration` (Run/Stop hold time)
- filter window sizes in `StateFilter`

## Troubleshooting

- “Cannot open camera”:
  - Close other apps using the webcam.
  - Check OS camera permission.
  - Try another camera index in `CameraWorker(camera_index=...)`.
- Slow first run:
  - DeepFace may download models. Set `DEEPFACE_HOME` to keep them cached.
- TensorFlow/oneDNN warnings:
  - They are noisy but harmless. If you really want silence:
    - `TF_ENABLE_ONEDNN_OPTS=0` (optional)

## Project Layout

- `main.py` — launches the PyQt5 app
- `src/qt_ui.py` — UI + camera thread + IPC to BF process
- `src/recognizer.py` — DeepFace analysis + position/vertical classification
- `src/filter.py` — smoothing + anti-jitter state filtering
- `src/state_machine.py` — hold-to-trigger logic and mapping
- `src/bf_worker_process.py` — Brainfuck execution process + tape updates

## Privacy

This project uses your webcam locally. No uploads, no cloud, no “AI overlord” dashboard.\
Just you, your face, and a language designed to hurt people.
