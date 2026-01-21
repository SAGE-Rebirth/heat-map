# HeatMap Vision - Real-Time AI Thermal Simulation

**HeatMap Vision** is a Python-based GUI application that turns a standard webcam into a simulated Thermal Camera. Using advanced **YOLO Segmentation**, **Skin Detection**, and **High-Intensity Thresholding**, it mimics physiological heat signatures for humans and animals, and detects potential fire/light sources.

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

### 🌡️ Physiological Heatmap
- **Humans**: Differentiates between clothes (Insulated/Warm) and exposed skin (Radiating/Hot).
- **Animals**: Supports detection for **Cats, Dogs, Birds, Horses, Bears, Sheep, and Cows**.

### 📏 Smart Distance 2.0
- **Adaptive Precision**: Automatically switches scaling logic based on what is visible.
    - **Face/Hand Mode**: Uses exposed skin size for accurate close-range measurements (< 1m).
    - **Body Mode**: Uses full bounding box height for walking subjects (> 2m).
- Detects subjects up to **10 meters** away.

### 🔥 Heat & Light Source Detection
- **Fire/Lights**: Detects high-intensity heat sources (light bulbs, lighters, fire) using brightness thresholding.
- **Visual Alert**: Highlights sources with a **Red Box** and `LIGHT SRC` tag.

### ⚡ Performance
- **Dual-Feed Display**: Side-by-side view (Raw + Thermal).
- **Optimized**: Threaded processing, frame skipping (Smart-FPS), and resolution scaling for smooth playback on standard CPUs.

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

*Note: The first run will automatically download the YOLOv8-Small model (detected as `yolo26s-seg.pt`).*

## Controls

*   **Start Camera**: Begins the video feed and AI processing.
*   **Stop Camera**: Stops the feed and releases camera resources.
*   **Q**: Press 'Q' on your keyboard to instantly quit the application.

## Technical Details

This project uses **Ultralytics YOLO** for multi-class instance segmentation and **OpenCV** for image processing.
For a deep dive into the V2.0 algorithms (Skin-Scaling Distance, animal masking, etc.), please refer to [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md).

## License

[MIT](LICENSE)
