import sys
import cv2
import torch
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QSlider, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QAbstractVideoSurface, QVideoFrame, QVideoSurfaceFormat
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QUrl

# Load YOLOv7 model for person detection
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

class CustomVideoSurface(QAbstractVideoSurface):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.widget = None
        
         # Initialize MediaPipe and OpenCV here
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def supportedPixelFormats(self, handleType):
        return [QVideoFrame.PixelFormat.Format_RGB32,
                QVideoFrame.PixelFormat.Format_ARGB32,
                QVideoFrame.PixelFormat.Format_BGR24]

    def start(self, format):
        self.format = format
        return super().start(format)
    
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

    def present(self, frame):
        if not frame.isValid():
            return False

        # Convert QVideoFrame to QImage
        image = frame.image()
        if image.isNull():
            return False
        
        # Convert QImage to OpenCV format
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        frame_array = np.array(ptr).reshape((height, width, 4))
        frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        modified_image = QImage(frame_rgb.data, width, height, QImage.Format.Format_RGB888)

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

        # Perform pose estimation using MediaPipe
        results_pose = self.pose.process(frame_rgb)

        # Draw skeleton if pose is detected
        if results_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame_rgb, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Update the widget with the modified frame
        if self.widget:
            pixmap = QPixmap.fromImage(modified_image)
            # Dynamically adjust QLabel size
            self.widget.setFixedSize(width, height)

            self.widget.setPixmap(pixmap)

        return True

    def setWidget(self, widget):
        self.widget = widget

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player with Frame Modification")
        self.setGeometry(200, 100, 800, 600)

        # Initialize media player
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoSurface = CustomVideoSurface()
        self.mediaPlayer.setVideoOutput(self.videoSurface)

        # QLabel to display modified frames
        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoSurface.setWidget(self.videoLabel)

        # UI Elements
        openButton = QPushButton("Open Video")
        openButton.clicked.connect(self.open_file)

        playButton = QPushButton("Play")
        playButton.clicked.connect(self.play_video)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.videoLabel)
        controls = QHBoxLayout()
        controls.addWidget(openButton)
        controls.addWidget(playButton)
        controls.addWidget(self.slider)
        layout.addLayout(controls)
        self.setLayout(layout)

        # Signal connections
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename != "":
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.mediaPlayer.play()

    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())