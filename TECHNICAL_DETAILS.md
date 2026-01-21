# Technical Details & Architecture

## Overview
This application is a real-time computer vision tool designed to simulate **Thermal Imaging** using standard RGB webcams. Unlike simple color filters, it employs **AI Segmentation** and **Physiological Logic** to create a biologically accurate representation of heat distribution on the human body.

## Core Technologies
- **Language**: Python 3.10+
- **GUI Framework**: PyQt6 (Thread-safe, non-blocking UI)
- **AI Engine**: Ultralytics YOLOv8/v26 (Segmentation Model)
- **Computer Vision**: OpenCV (cv2), NumPy

## The "Physiological" Heatmap Algorithm
The core innovation of this project is the **Skin-Aware Thermal Overlay**. It does not simply apply a color map to the detected object; it differentiates between "Insulated" and "Exposed" areas.

### 1. Instance Segmentation (YOLO)
We use `yolo26n-seg.pt` (Nano Segmentation Model) to detect humans.
- **Output**: A precise binary mask ($M_{body}$) defining the exact pixels of the person, excluding the background.
- **Benefit**: Ensures the "heat" does not spill onto walls or furniture, unlike bounding-box based methods.

### 2. Skin Detection (HSV Thresholding)
Inside the body mask, we apply color thresholding to identify exposed skin.
- **Input**: Converted HSV frame.
- **Ranges**:
  - Lower: `[0, 30, 60]`
  - Upper: `[20, 255, 255]`
- **Processing**:
  - A morphological "Open" operation removes noise.
  - Logical AND with $M_{body}$ creates the Skin Mask ($M_{skin}$).

### 3. Thermal Intensity Mapping
We construct a single-channel Grayscale Intensity Map ($I$) to represent temperature:
- **Background**: $I = 0$ (Cold/Black)
- **Clothes ($M_{body} - M_{skin}$)**: $I = 70$ (Warm/Purple/Red)
- **Skin ($M_{skin}$)**: $I = 255$ (Hot/Yellow/White)

### 4. Organic Blending
To prevent the result from looking like a cartoon labeling tool, we apply:
- **Gaussian Blur**: A heavy kernel (`31x31`) blends the sharp boundaries between clothes and skin. This simulates the natural radiation of heat from the body's core.
- **Colormap**: `cv2.applyColorMap(..., cv2.COLORMAP_INFERNO)` maps the intensities to a thermal spectrum (Black -> Purple -> Red -> Orange -> Yellow -> White).

## Distance Estimation
The application estimates the distance of the user from the camera using a **Pinhole Camera Model** approximation.
$$ Distance = \frac{\text{Real Height} \times \text{Focal Length}}{\text{Pixel Height}} $$
- **Real Height**: Assumed ~1.7m (Average).
- **Focal Length**: Approximated ~600px for standard 720p webcams.
- **Display**: Real-time overlay above the user's head.

## Performance Optimization
To ensure smooth 30+ FPS playback on standard hardware:
1.  **Threaded Architecture**: Video capture and AI inference run in a dedicated `QThread` (`VideoThread`), keeping the GUI responsive.
2.  **Inference Skipping**: AI runs every **3rd frame**. The application displays the *last generated heatmap* during gap frames, maintaining visual smoothness without overloading the GPU/CPU.
3.  **Resolution Scaling**: Input is resized to width=720px before processing to balance accuracy and speed.
4.  **Model Warmup**: A dummy inference is run at startup to load model weights into VRAM, preventing the first few seconds of video from stuttering.
