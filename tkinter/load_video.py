import sys
import cv2
import torch
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QStyle, QSizePolicy, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtCore import Qt, QUrl

# # Load YOLOv7 model for person detection
# model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Video Player")
        self.setGeometry(350, 100, 700, 500)
        # self.setWindowIcon(QIcon('player.png'))
        
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
        
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename != " ":
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.playBtn.setEnabled(True)
            
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
        
app = QApplication(sys.argv)
window = VideoPlayer()
sys.exit(app.exec_())

# # Path to the tennis video (replace with your actual video path)
# video_path = '/Users/yizhengc/Downloads/Sinner2.mp4'  # Change this to your actual video file path

# # Initialize MediaPipe pose detection
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils

# # Open the video file using OpenCV
# cap = cv2.VideoCapture(video_path)

# # Function to calculate IoU (Intersection over Union) between two bounding boxes
# def calculate_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     interWidth = max(0, xB - xA)
#     interHeight = max(0, yB - yA)
#     interArea = interWidth * interHeight

#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     unionArea = boxAArea + boxBArea - interArea

#     iou = interArea / unionArea if unionArea != 0 else 0
#     return iou

# def detect_contact():
#     # Check if the video file opened successfully
#     if not cap.isOpened():
#         print(f"Error: Could not open video file at {video_path}")
#         exit()

#     # Process video frame by frame
#     while True:
#         ret, frame = cap.read()

#         # Break the loop if the video ends
#         if not ret:
#             break

#         # Convert the frame (image) to a format that YOLOv7 can process
#         results = model(frame)

#         # Variables to store racket and ball bounding boxes
#         racket_box = None
#         ball_box = None

#         # Draw bounding boxes on the frame for detected objects
#         for det in results.xyxy[0]:  # For each detected object
#             xmin, ymin, xmax, ymax, conf, cls = det.tolist()

#             # Filter out low-confidence detections (confidence > 0.5)
#             if conf > 0.5:
#                 if model.names[int(cls)] == "tennis racket":
#                     racket_box = [xmin, ymin, xmax, ymax]
#                     cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
#                     cv2.putText(frame, f'Tennis Racket: {conf:.2f}', (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                 elif model.names[int(cls)] == "sports ball":
#                     ball_box = [xmin, ymin, xmax, ymax]
#                     cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
#                     cv2.putText(frame, f'Ball: {conf:.2f}', (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Check for overlap (contact) between racket and ball
#         if racket_box and ball_box:
#             print("Detect both")
#             iou = calculate_iou(racket_box, ball_box)
#             print("IoU is:", iou)
#             if iou > 0:  # Set an IoU threshold for detecting contact
#                 print("Contact Detected")

#         # Perform pose estimation using MediaPipe
#         results_pose = pose.process(frame)

#         # Draw skeleton if pose is detected
#         if results_pose.pose_landmarks:
#             mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         # Display the frame (optional, comment this out if not needed)
#         cv2.imshow('YOLOv7 + MediaPipe Skeleton Tracking', frame)

#         # Press 'q' to exit the video early
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture and writer objects
#     cap.release()

#     # Close any OpenCV windows
#     cv2.destroyAllWindows()

# detect_contact()