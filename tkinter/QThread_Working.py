import sys
import cv2
import torch
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QFileDialog, QSizePolicy, QStyle, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os

# Load YOLOv7 model for person detection
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.video_path = None
        
        # Initialize MediaPipe and OpenCV
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # YOLOv7 detection
                results = model(frame_rgb)
                
                # Process detections (same logic as before)
                contact_detected = False
                person_detections = []
                
                for det in results.xyxy[0]:
                    xmin, ymin, xmax, ymax, conf, cls = det.tolist()
                    if conf > 0.5:
                        label = model.names[int(cls)]
                        if label == "tennis racket":
                            cv2.rectangle(frame_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                        elif label == "sports ball":
                            cv2.rectangle(frame_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                        elif label == "person":
                            person_detections.append((xmin, ymin, xmax, ymax, conf))

                # Draw pose estimation
                results_pose = self.pose.process(frame_rgb)
                if results_pose.pose_landmarks:
                    self.mp_drawing.draw_landmarks(frame_rgb, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                self.change_pixmap_signal.emit(frame_rgb)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_video_path(self, path):
        self.video_path = path

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player with Frame Modification")
        self.setGeometry(200, 100, 800, 600)

        # Create the video thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)

        # Create the video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # UI Elements
        openButton = QPushButton("Open Video")
        openButton.clicked.connect(self.open_file)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play_video)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        controls = QHBoxLayout()
        controls.addWidget(openButton)
        controls.addWidget(self.playButton)
        layout.addLayout(controls)
        self.setLayout(layout)

        self._playing = False

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename:
            self.thread.set_video_path(filename)
            self.playButton.setEnabled(True)
            self.play_video()

    def play_video(self):
        if not self._playing:
            self.thread.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self._playing = True
        else:
            self.thread.stop()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self._playing = False

    def update_image(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale the image while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())