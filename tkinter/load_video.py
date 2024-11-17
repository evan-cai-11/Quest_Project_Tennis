import sys
import cv2
import torch
import mediapipe as mp
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QStyle, QSizePolicy, QFileDialog
from PyQt6.QtMultimedia import QMediaPlayer, QVideoSink
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QIcon, QPalette, QImage, QPixmap
from PyQt6.QtCore import Qt, QUrl, QTimer
import numpy as np

# Load YOLOv7 model for person detection
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.mediaPlayer = QMediaPlayer(self)
        
        # QVideoSink to capture frames
        self.video_sink = QVideoSink(self)
        self.mediaPlayer.setVideoOutput(self.video_sink)
        
        # Connect to capture each frame
        self.video_sink.videoFrameChanged.connect(self.detect_contact)
        
        # Display for modified frame
        self.modified_frame_label = QLabel(self)
        self.modified_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.setWindowTitle("Video Player")
        self.setGeometry(350, 100, 700, 500)
        
        # p = self.palette()
        # p.setColor(QPalette.window, Qt.black)
        # self.setPalette(p)
        
        # Initialize MediaPipe and OpenCV here
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.initUI()
        
        self.show()
        
        # Add initialization of video capture
        self.cap = None
        self.video_path = None
        
    def initUI(self):
        # self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()
        openBtn = QPushButton('Open Video')
        openBtn.clicked.connect(self.open_file)
        
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
        
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        
        hboxlayout = QHBoxLayout()
        hboxlayout.addWidget(openBtn)
        hboxlayout.addWidget(self.playBtn)
        hboxlayout.addWidget(self.slider)
        
        vboxlayout = QVBoxLayout()
        vboxlayout.addWidget(self.modified_frame_label)
        vboxlayout.addLayout(hboxlayout)
        
        self.setLayout(vboxlayout)
        
        self.mediaPlayer.setVideoOutput(videoWidget)
        
        self.mediaPlayer.playbackStateChanged.connect(self.media_state_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename != "":
            self.mediaPlayer.setSource(QUrl.fromLocalFile(filename))
            self.playBtn.setEnabled(True)
            self.video_path = filename
            self.cap = cv2.VideoCapture(filename)
            
    def play_video(self):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()
            
    # Function to calculate IoU (Intersection over Union) between two bounding boxes
    @staticmethod
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        unionArea = boxAArea + boxBArea - interArea

        iou = interArea / unionArea if unionArea != 0 else 0
        return iou
            
    def detect_contact(self, frame):
        # Convert QVideoFrame to QImage
        print("entering detection function")
        if frame.isValid():
            image = frame.toImage()
            if image.isNull():
                return
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print("finished converting frame")

            # Convert the frame (image) to a format that YOLOv7 can process
            results = model(frame_rgb)

            # Variables to store racket and ball bounding boxes
            racket_box = None
            ball_box = None

            # Draw bounding boxes on the frame for detected objects
            for det in results.xyxy[0]:  # For each detected object
                xmin, ymin, xmax, ymax, conf, cls = det.tolist()

                # Filter out low-confidence detections (confidence > 0.5)
                if conf > 0.5:
                    if model.names[int(cls)] == "tennis racket":
                        racket_box = [xmin, ymin, xmax, ymax]
                        cv2.rectangle(frame_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                        cv2.putText(frame_rgb, f'Tennis Racket: {conf:.2f}', (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    elif model.names[int(cls)] == "sports ball":
                        ball_box = [xmin, ymin, xmax, ymax]
                        cv2.rectangle(frame_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                        cv2.putText(frame_rgb, f'Ball: {conf:.2f}', (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Check for overlap (contact) between racket and ball
            if racket_box and ball_box:
                print("Detect both")
                iou = self.calculate_iou(racket_box, ball_box)
                print("IoU is:", iou)
                if iou > 0:  # Set an IoU threshold for detecting contact
                    print("Contact Detected")
                    self.mediaPlayer.pause()

            # Perform pose estimation using MediaPipe
            results_pose = self.pose.process(frame_rgb)

            # Draw skeleton if pose is detected
            if results_pose.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame_rgb, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
            pixmap = QPixmap.fromImage(frame_rgb)
            self.modified_frame_label.setPixmap(pixmap)
            
    def media_state_changed(self, state):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            
    def position_changed(self, position):
        self.slider.setValue(position)
        
    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
        
    def set_position(self, position):
        self.mediaPlayer.setPosition(position)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoPlayer()
    sys.exit(app.exec())