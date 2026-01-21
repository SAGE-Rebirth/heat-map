# Technical Details & Architecture

## Overview
This application is a real-time computer vision tool designed to simulate **Thermal Imaging** using standard RGB webcams. It uses **AI Segmentation** for biological targets (Humans, Animals) and **Computer Vision Thresholding** for high-intensity heat sources.

## Core Technologies
- **Language**: Python 3.10+
- **GUI Framework**: PyQt6 (Thread-safe, non-blocking UI)
- **AI Engine**: Ultralytics YOLOv8/v26 (Small Model: `yolo26s-seg.pt`)
- **Computer Vision**: OpenCV (cv2), NumPy

## 1. The "Physiological" Heatmap Algorithm
We process detected biological entities to simulate heat radiation.

### Instance Segmentation
We use `yolo26s-seg.pt` (Small Model) to generate binary masks for:
- **Classes**: Person (0), Bird (14), Cat (15), Dog (16), Horse (17), Sheep (18), Cow (19), Bear (21).
- **Benefit**: Confines the "heat" layer strictly to the subject's body.

### Skin Detection (HSV)
For Humans (Class 0), we further refine the mask to detect exposed skin:
- **Logic**: Exposed skin ( Face, Hands) radiates more heat than clothed areas.
- **Process**:
  1.  Convert frame to HSV.
  2.  Threshold for skin tones (`H:0-20`, `S:30-255`, `V:60-255`).
  3.  Morphological OPEN to remove noise.
  4.  Result: `valid_skin_mask`.

### Thermal Intensity Mapping ($I$)
- **Background**: $I = 0$ (Cold/Black)
- **Clothes/Fur**: $I = 70$ (Warm/Purple/Red)
- **Exposed Skin**: $I = 255$ (Hot/Yellow/White)
- **Blend**: Gaussian Blur (`31x31`) smooths transitions, simulating heat diffusion.
- **Color**: `cv2.applyColorMap(..., cv2.COLORMAP_INFERNO)`.

## 2. Adaptive Distance Estimation 2.0
Traditional single-camera distance estimation fails when a user is partially visible (e.g., just a hand). We use a **Context-Aware** approach.

### Formulas
The application approximates distance using the Pinhole Camera Model:
$$ D = \frac{R \times F}{P} $$
Where $R$ = Real Height, $F$ = Focal Length (~600px), $P$ = Pixel Height.

### Mode A: Skin-Based (Precision)
- **Trigger**: `if countNonZero(skin_mask) > Threshold`.
- **Target**: Largest contour in the skin mask (usually Face or Hand).
- **Constant**: $R \approx 20cm$ (Average Head/Hand Height).
- **Use Case**: Close-ups (0.2m - 1.0m).

### Mode B: Body-Based (General)
- **Trigger**: No significant skin mask found OR Non-Human Class.
- **Target**: Full Bounding Box.
- **Constant**: 
  - Human: $R \approx 170cm$
  - Animal: $R \approx 50cm$ (Rough average for pets).
- **Use Case**: Full body walking (1.5m - 10.0m).

## 3. High-Intensity Source Detection
Standard AI models do not detect "Heat" or generic "Fire". We use raw pixel intensity.
- **Method**: Global Thresholding.
- **Condition**: `Pixel_Value > 245` (Grayscale).
- **Process**: Find contours of these "blown out" bright spots.
- **Result**: Tagged as `LIGHT SRC | 0.99`. Detects light bulbs, the sun, lighters, and open flames.

## Performance Optimization
- **Model**: Switched from Nano (`n`) to Small (`s`) for better detection at >5m range.
- **Resolution**: Processing occurs at **960px width** (HD) to preserve distant details.
- **Asynchronous**: Inference runs in a separate `QThread`, decoupled from the Main GUI thread.
