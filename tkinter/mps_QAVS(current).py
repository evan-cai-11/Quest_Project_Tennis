import sys
import cv2
import torch
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QSlider, QFileDialog, QSizePolicy, QStyle, QComboBox, QMainWindow, QMessageBox
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
    def __init__(self, parent = None):
        super().__init__(parent)
        self.widget = None
        self.parent = parent

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

    def screenshot(self, person_box, frame_rgb):
        xmin, ymin, xmax, ymax, conf = person_box

        width = xmax - xmin
        height = ymax - ymin
                    
        CUSHION = 100
                    
        crop_xmin = int(xmin) - CUSHION
        crop_ymin = int(ymin) - CUSHION
        crop_xmax = int(xmax) + CUSHION
        crop_ymax = int(ymax) + CUSHION
                    
        frame_cropped = frame_rgb[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        file_name = "contact_screenshot.png"
        save_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent.screenshot_path = os.path.join(save_dir, file_name)
                    
        # Convert RGB to BGR for cv2.imwrite
        frame_bgr = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.parent.screenshot_path, frame_bgr)
        print(f"Screenshot saved to: {self.parent.screenshot_path}")
    
    def objectDetection(self, frame_rgb):
        # Convert the frame (image) to a format that YOLOv8 can process
        results = model(frame_rgb, device = "mps", verbose = False)[0]  # Get first result
        
        # Boolean Variables for contact, prep, and finish
        self.contact_detected = False
        self.prep_detected = False
        self.finish_detected = False

        # Variables to store racket, ball, and person bounding boxes
        racket_box = None
        ball_box = None
        person_box = None

        person_detections = []
        
        self.racket_right = None
        self.person_right = None
        self.racket_left = None
        self.person_left = None
        

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
            if iou > 0:
                self.parent.contact_detected = True 
                print("Contact Detected")
                
                if person_box:
                    self.screenshot(person_box, frame_rgb)

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

class ComparisonWindow(QWidget):
    def __init__(self, screenshot_path):
        super().__init__()
        self.setWindowTitle("Pro Comparison")
        self.setGeometry(200, 100, 1000, 600)

        layout = QVBoxLayout()
        
        comparison_layout = QHBoxLayout()

        rafa_contact_path = "/Users/yizhengc/dev/Quest_Project_Tennis/images/rafa_forehand_contact_ao.png"
        
        screenshot_label = QLabel(self)
        comparison_photo_label = QLabel(self)
        if screenshot_path and os.path.exists(screenshot_path):
            screenshot = QPixmap(screenshot_path)
            screenshot_label.setPixmap(screenshot.scaled(400, 400, Qt.KeepAspectRatio))
            comparison_photo = QPixmap(rafa_contact_path)
            comparison_photo_label.setPixmap(comparison_photo.scaled(400, 400, Qt.KeepAspectRatio))
            
        comparison_layout.addWidget(screenshot_label)
        comparison_layout.addWidget(comparison_photo_label)
        
        layout.addLayout(comparison_layout)
        self.setLayout(layout)

class VideoPlayer(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.contact_detected = False 
        self.screenshot_path = None
        self.parent = parent

        rafa_chosen = False
        sincity_chosen = False

        self.setWindowTitle("Video Player with Frame Modification")
        self.setGeometry(200, 100, 800, 600)

        # Initialize media player
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoSurface = CustomVideoSurface(self)  # Pass self as parent
        self.mediaPlayer.setVideoOutput(self.videoSurface)

        # QLabel to display modified frames
        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setMinimumSize(640, 480)  # Set minimum size
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoSurface.setWidget(self.videoLabel)

        # UI Elements
        self.openButton = QPushButton("Open Video")
        self.openButton.clicked.connect(self.open_file)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play_video)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)

        # Signal connections
        self.mediaPlayer.stateChanged.connect(self.media_state_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.setup)

        self.dropdown = QComboBox(self)
        self.dropdown.addItem("Rafa")
        self.dropdown.addItem("SinCity")

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next)

        self.dropdown_chosen = QPushButton("Get Comparison Photo")
        self.dropdown_chosen.clicked.connect(self.dropdown_choice)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.videoLabel)
        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.dropdown)
        controls.addWidget(self.dropdown_chosen)
        self.layout.addLayout(controls)
        self.setLayout(self.layout)

        self.page = 0  # Add page tracking

    def setup(self):
        controls = QHBoxLayout()
        self.start_button.setParent(None)
        self.dropdown.setParent(None)
        controls.addWidget(self.openButton)
        controls.addWidget(self.playButton)
        controls.addWidget(self.slider)
        controls.addWidget(self.next_button)
        self.layout.addLayout(controls)
        
        self.setLayout(self.layout)
        self.videoSurface.setWidget(self.videoLabel)
        
        self.page = 1

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename != "":
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.playButton.setEnabled(True)
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
        self.playButton.setEnabled(True)

        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setMinimumSize(640, 480)
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoSurface.setWidget(self.videoLabel)

        self.mediaPlayer.play()
        
    def current_text_via_index(self, index):
        ctext = self.combobox.itemText(index)
        print("Current itemText", ctext)

    def dropdown_choice(self):
        chosen = self.dropdown.currentText()
        if chosen == "Rafa":
            self.rafa_chosen = True
        elif chosen == "SinCity":
            self.sincity_chosen = True
        else:
            msg = QMessageBox()
            msg.setText("Please select a player to compare with")
            msg.exec_()

    def next(self):
        if self.page == 1 and self.contact_detected:
            self.comparison_window = ComparisonWindow(self.screenshot_path)
            self.comparison_window.show()
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())