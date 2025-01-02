import sys
import cv2
import torch
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QSlider, QFileDialog, QSizePolicy, QStyle, QComboBox, QMainWindow
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QAbstractVideoSurface, QVideoFrame, QVideoSurfaceFormat
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QUrl, QTimer
import os
from ultralytics import YOLO

# Load YOLOv8 model for person detection
model = YOLO("yolov8m")

class CustomVideoSurface(QAbstractVideoSurface):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.widget = None

        # Initialize MediaPipe and OpenCV
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
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

    def present_do_nothing(self, frame):
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
    
    def objectDetection(self, frame_rgb):
        # Convert the frame (image) to a format that YOLOv8 can process
        results = model(frame_rgb, device="mps")[0]  # Get first result
        
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
        for det in results:  # For each detected object
            xmin, ymin, xmax, ymax = det.boxes.xyxy[0].tolist()  # Get box coordinates
            conf = det.boxes.conf[0].item()  # Get confidence
            cls = det.boxes.cls[0].item()  # Get class id

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
                # Generate a filename with sequential numbering
                file_name = f"contact_screenshot.png"
                
                # Define the directory where you want to save the screenshot
                save_dir = os.path.expanduser("~")
                file_path = os.path.join(save_dir, file_name)

                # Save the screenshot
                frame_rgb.image().save(file_path)
                
        # Check for preparation 
        # if racket_box and ball_box and len(self.frames) > 1:
        #     if racket_left > person_left:
        #         print("Backhand Preparation Detected")
        #     elif racket_right > person_right:
        #         print("Forehand Preparation Detected")


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

        self.objectDetection(frame_rgb)

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

    def setWidget(self, widget):
        self.widget = widget

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()

        # self.start()

        self.setWindowTitle("Video Player with Frame Modification")
        self.setGeometry(200, 100, 800, 600)

        # Initialize media player
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoSurface = CustomVideoSurface()
        self.mediaPlayer.setVideoOutput(self.videoSurface)

        # QLabel to display modified frames
        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setMinimumSize(640, 480)  # Set minimum size
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoSurface.setWidget(self.videoLabel)

        # UI Elements
        openButton = QPushButton("Open Video")
        openButton.clicked.connect(self.open_file)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play_video)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.videoLabel)
        controls = QHBoxLayout()
        controls.addWidget(openButton)
        controls.addWidget(self.playButton)
        controls.addWidget(self.slider)
        layout.addLayout(controls)
        self.setLayout(layout)

        # Signal connections
        self.mediaPlayer.stateChanged.connect(self.media_state_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        
        start_button = QPushButton("Start")
        start_button.setFixedSize(200, 60)
        start_button.clicked.connect(self.start)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename != "":
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.playButton.setEnabled(True)  # Enable the play button
            self.mediaPlayer.play()

    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()
            
    def media_state_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def load_new_video(self, filename):
        self.mediaPlayer.stop()
        self.videoSurface.setWidget(None)
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
        self.playButton.setEnabled(True)  # Enable the play button

        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setMinimumSize(640, 480)
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoSurface.setWidget(self.videoLabel)

        self.mediaPlayer.play()

    def dropdown(self):
        combobox = QComboBox(self)
        combobox.addItem("Rafa")
        combobox.addItem("Sinner")
        combobox.move(50, 50)
        
    def current_text_via_index(self, index):
        ctext = self.combobox.itemText(index)
        print("Current itemText", ctext)

    def start(self):
        self.videoSurface.setWidget(None)
        main_window = QMainWindow()
        main_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())