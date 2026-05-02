import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, 
    QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QFrame
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import pyqtSlot, Qt
import logging
from video_procs import VideoThread

logger = logging.getLogger("GUI")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HeatMap Vision - YOLO26")
        self.resize(1280, 720)
        
        # Styles
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005f9e;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #aaaaaa;
            }
        """)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout
        self.main_layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("Real-time Vision Analytics: Standard vs Heatmap")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(header)

        # Video Area (Side by side)
        video_layout = QHBoxLayout()
        
        # Left Feed (Standard)
        self.left_panel = self.create_video_panel("Live Feed (Standard)")
        self.left_label = self.left_panel.findChild(QLabel, "video_label")
        video_layout.addWidget(self.left_panel)
        
        # Right Feed (Heatmap)
        self.right_panel = self.create_video_panel("AI Heatmap (YOLO26)")
        self.right_label = self.right_panel.findChild(QLabel, "video_label")
        video_layout.addWidget(self.right_panel)
        
        self.main_layout.addLayout(video_layout)

        # Controls
        control_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Camera")
        self.btn_start.clicked.connect(self.start_video)
        
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)
        
        control_layout.addStretch()
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch()
        
        self.main_layout.addLayout(control_layout)

        # Video Thread
        self.thread = None

    def create_video_panel(self, title_text):
        """Creates a styled video panel with a title."""
        frame = QFrame()
        frame.setStyleSheet("background-color: #2d2d2d; border-radius: 10px;")
        layout = QVBoxLayout(frame)
        
        title = QLabel(title_text)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        video_label = QLabel()
        video_label.setObjectName("video_label")
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_label.setStyleSheet("background-color: #000000; border-radius: 5px;")
        video_label.setMinimumSize(640, 480)
        layout.addWidget(video_label)
        
        return frame

    def start_video(self):
        logger.info("User requested to start video.")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        # Initialize thread
        self.thread = VideoThread()
        self.thread.raw_frame_signal.connect(self.update_left_feed)
        self.thread.heatmap_frame_signal.connect(self.update_right_feed)
        self.thread.error_signal.connect(self.show_error)
        self.thread.start()

    def stop_video(self):
        logger.info("User requested to stop video.")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.left_label.setText("Stopping...")
        self.right_label.setText("Stopping...")
        
        if self.thread and self.thread.isRunning():
            # Connect finish signal to cleanup
            # We use UniqueConnection to prevent multiple connections if clicked fast
            try:
                self.thread.finished.disconnect(self.on_thread_finished)
            except TypeError:
                pass # Not connected
                
            self.thread.finished.connect(self.on_thread_finished)
            self.thread.stop() # This now just signals, we removed .wait() in the Plan for video_procs
        else:
            self.on_thread_finished()

    def on_thread_finished(self):
        logger.info("Video thread finished.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        # clear labels
        self.left_label.clear()
        self.right_label.clear()
        self.left_label.setText("Camera Stopped")
        self.right_label.setText("AI Stopped")
        self.thread = None

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Q:
            logger.info("'Q' pressed. Exiting application.")
            self.close()

    @pyqtSlot(np.ndarray)
    def update_left_feed(self, frame):
        self.display_frame(self.left_label, frame)

    @pyqtSlot(np.ndarray)
    def update_right_feed(self, frame):
        self.display_frame(self.right_label, frame)
        
    def display_frame(self, label_widget, frame):
        """Converts CV2 Frame to QPixmap and displays it."""
        if frame is None:
            return
            
        # CV2 is BGR, PyQt needs RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit label, keep aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            label_widget.width(), label_widget.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label_widget.setPixmap(scaled_pixmap)

    @pyqtSlot(str)
    def show_error(self, message):
        logger.error(f"GUI Error Alert: {message}")
        QMessageBox.critical(self, "Error", message)
        self.stop_video()

    def closeEvent(self, event):
        self.stop_video()
        event.accept()
