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
        
        self.second_to_last_overlap = False
        self.previous_overlap = False

        self.screenshots_number = 1

        # Initialize MediaPipe and OpenCV
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
    def start(self, format):
        self.format = format
        return super().start(format)

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

        if (boxAArea > boxBArea):
            iou = interArea / boxBArea if boxBArea != 0 else 0
        else: 
            iou = interArea / boxAArea if boxAArea != 0 else 0

        return iou

    def screenshot(self, person_box, racket_box, frame_rgb):
        person_xmin, person_ymin, person_xmax, person_ymax, _ = person_box
        racket_xmin, racket_ymin, racket_xmax, racket_ymax = racket_box
                
        CUSHION = 60
        
        crop_xmin = int(min(person_xmin, racket_xmin)) - CUSHION
        crop_ymin = int(min(person_ymin, racket_ymin)) - CUSHION
        crop_xmax = int(max(person_xmax, racket_xmax)) + CUSHION
        crop_ymax = int(max(person_ymax, racket_ymax)) + CUSHION
                
        frame_cropped = frame_rgb[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        file_name = f"contact_screenshot{self.screenshots_number}.png"
        save_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent.screenshot_path = os.path.join(save_dir, file_name)
        frame_bgr = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.parent.screenshot_path, frame_bgr)
        print(f"Screenshot saved to: {self.parent.screenshot_path}")

        self.screenshots_number += 1
    
    def objectDetection(self, frame_rgb):
        # Convert the frame (image) to a format that YOLOv8 can process
        results = model(frame_rgb, device = "mps", verbose = False)[0]
        
        self.contact_detected = False
        self.prep_detected = False
        self.finish_detected = False

        racket_box = None
        ball_box = None
        person_box = None

        person_detections = []
        
        self.racket_right = None
        self.person_right = None
        self.racket_left = None
        self.person_left = None
        

        # Draw the bounding boxes
        for det in results:
            xmin, ymin, xmax, ymax = det.boxes.xyxy[0].tolist()
            conf = det.boxes.conf[0].item()
            cls = det.boxes.cls[0].item()

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

        # Check for overlap between racket and ball
        if racket_box and ball_box:
            iou_ball = self.calculate_iou(racket_box, ball_box)
            iou_racket = self.calculate_iou(racket_box, person_box)

            print("IoU_Ball: ", iou_ball)
            print("IoU_Racket: ", iou_racket)
            
            if iou_ball > 0.5 and iou_racket < 0.5:
                self.parent.contact_detected = True
                
                if person_box:
                    self.screenshot(person_box, racket_box, frame_rgb)
            
        #     self.second_to_last_overlap = self.previous_overlap
        #     self.previous_overlap = current_overlap
        # else:
        #     self.second_to_last_overlap = False
        #     self.previous_overlap = False

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
    def __init__(self, screenshot_path, pro_player, parent = None):
        super().__init__(parent)
        self.setGeometry(200, 100, 1000, 600)

        self.parent = parent
        self.screenshot_path = screenshot_path

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        self.screenshot_paths = []

        layout = QVBoxLayout()
        comparison_layout = QHBoxLayout()

        self.user_dropdown = QComboBox(self)
        self.user_dropdown.currentTextChanged.connect(self.update_user_photo)

        self.pro_dropdown = QComboBox(self)
        self.pro_dropdown.addItem("Rafa")
        self.pro_dropdown.addItem("Sinner")
        self.pro_dropdown.currentTextChanged.connect(self.update_pro_photo)

        self.screenshot_label = QLabel(self)
        self.comparison_photo_label = QLabel(self)

        self.update_screenshot(self.screenshot_path)
        self.update_pro_photo(pro_player)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.user_dropdown)
        controls_layout.addWidget(self.pro_dropdown)

        comparison_layout.addWidget(self.screenshot_label)
        comparison_layout.addWidget(self.comparison_photo_label)
        comparison_layout.addLayout(controls_layout)
        
        layout.addLayout(comparison_layout)
        self.setLayout(layout)

    def update_screenshot(self, path):
        self.screenshot_paths.append(path)
        self.user_dropdown.addItem(f"Screenshot {len(self.screenshot_paths)}")
        self.update_user_photo(f"Screenshot {len(self.screenshot_paths)}")

    def update_user_photo(self, selection):
        user_photos = {
            "Screenshot 1": "/Users/yizhengc/dev/Quest_Project_Tennis/tkinter/contact_screenshot1.png",
            "Screenshot 2": "/Users/yizhengc/dev/Quest_Project_Tennis/tkinter/contact_screenshot2.png"
        }

        user_path = user_photos.get(selection)
        if user_path and os.path.exists(user_path):
            user_photo = QPixmap(user_path)
            self.screenshot_label.setPixmap(user_photo.scaled(400, 400, Qt.KeepAspectRatio))
        else:
            self.screenshot_label.clear()

    def update_pro_photo(self, player):
        pro_photos = {
            "Rafa": "/Users/yizhengc/dev/Quest_Project_Tennis/images/rafa_forehand_contact_ao.png",
            "Sinner": "/Users/yizhengc/dev/Quest_Project_Tennis/images/sinner_forehand_contact.png"
        }
        
        pro_path = pro_photos.get(player)
        if pro_path and os.path.exists(pro_path):
            comparison_photo = QPixmap(pro_path)
            self.comparison_photo_label.setPixmap(comparison_photo.scaled(400, 400, Qt.KeepAspectRatio))
        else:
            self.comparison_photo_label.clear()

class VideoPlayer(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.contact_detected = False 
        self.screenshot_path = None
        self.parent = parent

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

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.videoLabel)
        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        self.layout.addLayout(controls)
        self.setLayout(self.layout)

        self.page = 0 
        self.selected_player = "Rafa" # default

    def setup(self):
        controls = QHBoxLayout()
        self.start_button.setParent(None)
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

    def next(self) :
        if self.page == 1 and self.contact_detected:
            self.comparison_window = ComparisonWindow(self.screenshot_path, self.selected_player)
            self.comparison_window.show()
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())