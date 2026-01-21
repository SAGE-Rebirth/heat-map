# HeatMap Vision - Real-Time AI Thermal Simulation

**HeatMap Vision** is a Python-based GUI application that turns a standard webcam into a simulated Thermal Camera. Using advanced **YOLO Segmentation** and **Skin Detection** logic, it creates a physiologically accurate heat map where exposed skin (face, hands) appears "hottest," while clothed areas appear "warm," distinct from the cold background.

![Preview Placeholder](https://via.placeholder.com/800x400?text=HeatMap+Vision+Preview)

---

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
    - [Windows](#windows)
    - [macOS](#macos)
    - [Linux](#linux)
- [How to Run](#how-to-run)
- [Controls](#controls)
- [Technical Details](#technical-details)
- [License](#license)

---

## Features

*   **Dual-Feed Display**: Side-by-side view of the standard raw feed and the AI-augmented thermal feed.
*   **Physiological Accuracy**: Differentiates between clothes (Insulated/Warm) and skin (Radiating/Hot).
*   **Real-Time Segmentation**: Heat overlay conforms strictly to the human body shape, ignoring background clutter.
*   **Distance Estimation**: Estimates and displays the user's distance from the camera in real-time.
*   **Performance Optimized**: Uses threaded processing and frame skipping to maintain a responsive UI even during heavy inference.
*   **Smart Warmup**: Pre-loads AI models to prevent startup lag.

## Prerequisites

- **Python 3.8** or higher.
- A working **Webcam**.
- GPU recommended (NVIDIA CUDA) but works efficiently on CPU.

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Start-ignite/Heat-Map-Video-Feed-Gui.git
cd Heat-Map-Video-Feed-Gui
```

### Windows
```powershell
# Create Virtual Environment
python -m venv venv

# Activate Environment
.\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### macOS
```bash
# Install Python (if not installed)
brew install python

# Create Virtual Environment
python3 -m venv venv

# Activate Environment
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### Linux
```bash
# Install Python venv (Debian/Ubuntu)
sudo apt update && sudo apt install python3-venv

# Create Virtual Environment
python3 -m venv venv

# Activate Environment
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

## How to Run

Once dependencies are installed and the virtual environment is active:

```bash
python main.py
```

The application window will open. Click **"Start Camera"** to begin.

*Note: The first run may take a few seconds to download the YOLO model weights.*

## Controls

*   **Start Camera**: Begins the video feed and AI processing.
*   **Stop Camera**: Stops the feed and releases camera resources.
*   **Q**: Press 'Q' on your keyboard to instantly quit the application.

## Technical Details

This project uses **Ultralytics YOLOv8/v26** for instance segmentation and **OpenCV** for image processing.
For a deep dive into the heatmap algorithm, skin detection logic, and distance estimation formulas, please refer to [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md).

## License

[MIT](LICENSE)
