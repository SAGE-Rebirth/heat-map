import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from ultralytics import YOLO
import time
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger("VideoBackend")

class VideoThread(QThread):
    """
    Thread for capturing video and running YOLO heatmap inference.
    Emits signals for both the raw frame and the heatmap overlay.
    """
    
    # Signals to update the UI
    # We send numpy arrays (BGR format) which the GUI will convert to QPixmap
    raw_frame_signal = pyqtSignal(np.ndarray)
    heatmap_frame_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, camera_index=0, model_path="yolo26s-seg.pt"):
        super().__init__()
        self.camera_index = camera_index
        self.model_path = model_path
        self._is_running = False
        
        # Performance monitoring
        self.prev_time = 0
        self.fps = 0

    def run(self):
        logger.info(f"Starting VideoThread with camera index {self.camera_index}")
        self._is_running = True
        
        # Open Camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("Failed to open camera!")
            self.error_signal.emit("Could not open video camera. Please check connections.")
            return

        if not self._is_running or self.isInterruptionRequested():
            cap.release()
            return

        # Optimization Parameters
        target_width = 960 # Higher resolution (HD) for better long-range detection
        inference_interval = 3 
        frame_count = 0
        last_heatmap_frame = None

        # Load YOLO Segmentation Model
        try:
            logger.info(f"Loading YOLO model: {self.model_path}")
            model = YOLO(self.model_path)
            
            # WARMUP
            logger.info("Running warmup inference...")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = model(dummy_frame, verbose=False, classes=[0], retina_masks=True)
            logger.info("YOLO Segmentation Model loaded and warmed up.")
            
        except Exception as e:
            logger.critical(f"Failed to initialize YOLO model: {e}")
            self.error_signal.emit(f"AI Model Error: {str(e)}")
            cap.release()
            return
            
        if not self._is_running or self.isInterruptionRequested():
            cap.release()
            return
        
        while self._is_running and cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to read frame from camera.")
                time.sleep(0.1)
                continue

            # Resize frame
            h, w = frame.shape[:2]
            aspect_ratio = h / w
            new_w = target_width
            new_h = int(new_w * aspect_ratio)
            frame = cv2.resize(frame, (new_w, new_h))

            # Calculate FPS
            curr_time = time.time()
            fps_val = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = curr_time
            
            # --- 1. Raw Feed ---
            self.raw_frame_signal.emit(frame.copy())
            
            # --- 2. Heatmap Feed (True Thermal via Segmentation) ---
            frame_count += 1
            
            if frame_count % inference_interval == 0 or last_heatmap_frame is None:
                try:
                    # Run Inference (Retina masks for better quality)
                    # Lower confidence threshold efficiently allows distant detection
                    results = model(frame, verbose=False, classes=[0], conf=0.25, retina_masks=True)
                    
                    thermal_overlay = None

                    if results and results[0].masks is not None:
                        # Extract Masks (N, H, W) -> float32 on GPU usually
                        # Convert to numpy and combine all masks (logical OR)
                        # We use max to combine overlapping masks
                        masks = results[0].masks.data.cpu().numpy() # Shape (N, H, W)
                        
                        # Resize masks to match frame if needed (sometimes YOLO returns varying sizes)
                        # Actually 'retina_masks=True' tries to match orig img shape but resized frame input might vary.
                        # It's safest to resize the final mask to the frame size.
                        
                        full_mask = np.zeros((new_h, new_w), dtype=np.uint8)

                        for mask in masks:
                            # Mask is float 0-1. Resize to frame size.
                            # Note: masks.data might be 640x640 (model size). 
                            # We need to resize it to our 720x405 frame.
                            resized_mask = cv2.resize(mask, (new_w, new_h))
                            # Binarize
                            binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255
                            full_mask = cv2.bitwise_or(full_mask, binary_mask)

                        if np.count_nonzero(full_mask) > 0:
                            # --- Physiologically Accurate Thermal Simulation ---
                            # Logic: 
                            # 1. Clothes insulate heat -> Appear Warm (Red/Purple/Orange)
                            # 2. Exposed Skin emits heat -> Appears Hot (Yellow/White)
                            
                            # A. Skin Detection (HSV)
                            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            # Standard HSV range for skin tones
                            lower_skin = np.array([0, 30, 60], dtype=np.uint8)
                            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                            
                            raw_skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
                            
                            # Refine Skin Mask: Morphological Open to remove noise
                            kernel = np.ones((3, 3), np.uint8)
                            raw_skin_mask = cv2.morphologyEx(raw_skin_mask, cv2.MORPH_OPEN, kernel)
                            
                            # Only consider skin that is INSIDE the person segmentation
                            valid_skin_mask = cv2.bitwise_and(raw_skin_mask, full_mask)
                            
                            # B. Build Heat Intensity Map (Grayscale)
                            heat_intensity = np.zeros((new_h, new_w), dtype=np.uint8)
                            
                            # Level 1: Body Heat (Covered by Clothes) -> Low Intensity (e.g., 70/255)
                            # This will map to Purple/Red in INFERNO
                            heat_intensity[full_mask > 0] = 70 
                            
                            # Level 2: Face/Hands Heat (Exposed) -> High Intensity (255/255)
                            # This will map to Yellow/White in INFERNO
                            heat_intensity[valid_skin_mask > 0] = 255
                            
                            # Level 3: Smooth transitions
                            # Blur the intensity map so the "hot" skin radiates slightly into the clothes
                            heat_intensity = cv2.GaussianBlur(heat_intensity, (31, 31), 0)
                            
                            # C. Colorize
                            thermal_colors = cv2.applyColorMap(heat_intensity, cv2.COLORMAP_INFERNO)
                            
                            # D. Masking
                            # Re-apply the sharp person mask so the heat doesn't spill into the background
                            thermal_colors = cv2.bitwise_and(thermal_colors, thermal_colors, mask=full_mask)
                            
                            # E. Blend
                            # High opacity for thermal look, low opacity for background context
                            thermal_overlay = cv2.addWeighted(frame, 0.4, thermal_colors, 0.6, 0)
                        else:
                            thermal_overlay = frame
                    else:
                        thermal_overlay = frame

                    # Ensure contiguous
                    if not thermal_overlay.flags['C_CONTIGUOUS']:
                        thermal_overlay = np.ascontiguousarray(thermal_overlay)
                    
                    # --- Distance Estimation and Overlay (For ALL detected persons) ---
                    if results and len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            box_h = y2 - y1
                            
                            if box_h > 0:
                                known_height_cm = 170  # Average human height
                                focal_length_approx = 600 # Approximate focal length
                                distance_cm = (known_height_cm * focal_length_approx) / box_h
                                distance_m = distance_cm / 100
                                
                                # Draw Distance Label on thermal_overlay
                                cv2.putText(thermal_overlay, f"Dist: {distance_m:.1f}m", (x1, y1 - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # Add static max range overlay
                    cv2.putText(thermal_overlay, f"Max Range: ~10.0m", (20, new_h - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    last_heatmap_frame = thermal_overlay

                except Exception as e:
                    if int(curr_time) % 5 == 0:
                        logger.error(f"Inference error: {e}")
                    last_heatmap_frame = frame 
            
            if last_heatmap_frame is not None:
                display_frame = last_heatmap_frame.copy()
                cv2.putText(display_frame, f"FPS: {int(fps_val)}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.heatmap_frame_signal.emit(display_frame)
            else:
                self.heatmap_frame_signal.emit(frame)

            time.sleep(0.005)
            
            if self.isInterruptionRequested():
                 break

        logger.info("Releasing camera and stopping thread.")
        cap.release()

    def stop(self):
        """Stops the thread safely without blocking."""
        self._is_running = False
        self.requestInterruption()
