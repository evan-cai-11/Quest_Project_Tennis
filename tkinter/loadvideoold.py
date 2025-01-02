import sys
import cv2
import torch
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QStyle, QSizePolicy, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QIcon, QPalette, QImage, QPixmap
from PyQt5.QtCore import Qt, QUrl, QTimer

# Load YOLOv7 model for person detection
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Video Player")
        self.setGeometry(350, 100, 700, 500)
        
        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)
        
        self.initUI()
        
        self.show()
        
    def initUI(self):
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()
        openBtn = QPushButton('Open Video')
        openBtn.clicked.connect(self.open_file)
        
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
        
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        
        hboxlayout = QHBoxLayout()
        hboxlayout.addWidget(openBtn)
        hboxlayout.addWidget(self.playBtn)
        hboxlayout.addWidget(self.slider)
        
        vboxlayout = QVBoxLayout()
        vboxlayout.addWidget(videoWidget)
        vboxlayout.addLayout(hboxlayout)
        
        self.setLayout(vboxlayout)
        
        self.mediaPlayer.setVideoOutput(videoWidget)
        
        self.mediaPlayer.stateChanged.connect(self.media_state_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize OpenCV VideoCapture (empty for now)
        self.cap = None
        
        # Timer for detecting contact every frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.present)
    
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename:
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.mediaPlayer.play()
            
            # Initialize OpenCV capture
            self.cap = cv2.VideoCapture(filename)
            self.playBtn.setEnabled(True)
            self.timer.start(16)  # Check for contact every 30 ms
            
    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()
            
    def media_state_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            
    def position_changed(self, position):
        self.slider.setValue(position)
        
    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
        
    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def calculate_iou(self, boxA, boxB):
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

        return interArea / unionArea if unionArea != 0 else 0
    
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
           
        # Boolean Variables for contact, prep, and finish
        contact_detected = False
        prep_detected = False
        finish_detected = False

        # Variables to store racket, ball, and person bounding boxes
        racket_box = None
        ball_box = None
        person_box = None

        person_detections = []
        
        racket_right = None
        person_right = None
        racket_left = None
        person_left = None

        # Draw bounding boxes on the frame for detected objects
        for det in results.xyxy[0]:  # For each detected object
            xmin, ymin, xmax, ymax, conf, cls = det.tolist()

            # Filter out low-conf detections
            if conf > 0.5:
                label = model.names[int(cls)]
                if label == "tennis racket":
                    racket_box = [xmin, ymin, xmax, ymax]
                    cv2.rectangle(frame_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    cv2.putText(frame_rgb, f'Tennis Racket: {conf:.2f}', (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    racket_right = xmax
                    racket_left = xmin
                elif label == "sports ball":
                    ball_box = [xmin, ymin, xmax, ymax]
                    cv2.rectangle(frame_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                    cv2.putText(frame_rgb, f'Ball: {conf:.2f}', (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif label == "person":
                    person_detections.append((xmin, ymin, xmax, ymax, conf))
                    person_right = xmax
                    person_left = xmin

        if person_detections:
            person_box = max(person_detections, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))

        if person_box:
            xmin, ymin, xmax, ymax, conf = person_box
            cv2.rectangle(frame_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
            cv2.putText(frame_rgb, f'Person: {conf:.2f}', (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Check for overlap (contact) between racket and ball
        if racket_box and ball_box:
            iou = self.calculate_iou(racket_box, ball_box)
            print("IoU: ", iou)
            if iou > 0:  # Set an IoU threshold for detecting contact
                contact_detected = True
                print("Contact Detected")
                
        # Check for preparation 
        # if racket_box and ball_box and len(self.frames) > 1:
        #     if racket_left > person_left:
        #         print("Backhand Preparation Detected")
        #     elif racket_right > person_right:
        #         print("Forehand Preparation Detected")

        # Perform pose estimation using MediaPipe
        results_pose = self.pose.process(frame_rgb)

        # Draw skeleton if pose is detected
        if results_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame_rgb, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Update the widget with the modified frame
        if self.widget:
            pixmap = QPixmap.fromImage(modified_image)
            
            # Calculate scaled size while maintaining aspect ratio
            widget_size = self.widget.size()
            scaled_pixmap = pixmap.scaled(widget_size, 
                                        Qt.KeepAspectRatio, 
                                        Qt.SmoothTransformation)
            # Center the pixmap in the label
            self.widget.setPixmap(scaled_pixmap)
            
        return True
    
app = QApplication(sys.argv)
window = VideoPlayer()
sys.exit(app.exec_())