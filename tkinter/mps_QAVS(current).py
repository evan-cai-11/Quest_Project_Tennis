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

model = YOLO("yolov8m")

class CustomVideoSurface(QAbstractVideoSurface):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.widget = None
        self.parent = parent
        
        self.second_to_last_overlap = False
        self.previous_overlap = False
        self.current_overlap = True

        self.frames = []
        self.size = 0

        self.screenshots_number = 1

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        self.position_history = []
        
        self.backhand_prep = False
        self.forehand_prep = False

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
        
        for det in results:
            xmin, ymin, xmax, ymax = det.boxes.xyxy[0].tolist()
            conf = det.boxes.conf[0].item()
            cls = det.boxes.cls[0].item()

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

        if racket_box and ball_box:
            iou_ball = self.calculate_iou(racket_box, ball_box)
            iou_racket = self.calculate_iou(racket_box, person_box)

            print("IoU_Ball: ", iou_ball)
            print("IoU_Racket: ", iou_racket)
            
            if iou_ball > 0.5 and iou_racket < 0.5:
                self.parent.contact_detected = True
                
                if person_box:
                    self.screenshot(person_box, racket_box, frame_rgb)

            # if racket_box and ball_box:
            #     if racket_left > person_left and self.forehand_prep == True:
            #         print("Forehand Finish Detected")
            #     elif racket_right > person_right and self.backhand_prep == True:
            #         print("Backhand Finish Detected")

            # if racket_box and person_box:
            #     racket_left = racket_box[0]
            #     person_left = person_box[0]
                
            #     self.position_history.append((racket_left, person_left))
                
            #     if len(self.position_history) >= 3:
            #         third_to_last = self.position_history[-3]
            #         racket_left_past, person_left_past = third_to_last
                    
            #         if racket_left_past < person_left_past:
            #             self.backhand_prep = True
            #         elif racket_left_past > person_left_past:
            #             self.forehand_prep = True

            # self.frames.append(frame_rgb)
            # self.size += 1

    def present(self, frame):
        if not frame.isValid():
            return False

        image = frame.image()
        if image.isNull():
            return False

        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        frame_array = np.array(ptr).reshape((height, width, 4))
        frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        modified_image = QImage(frame_rgb.data, width, height, QImage.Format.Format_RGB888)

        self.objectDetection(frame_rgb)

        results_pose = self.pose.process(frame_rgb)

        if results_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame_rgb, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        if self.widget:
            pixmap = QPixmap.fromImage(modified_image)
            
            widget_size = self.widget.size()
            scaled_pixmap = pixmap.scaled(widget_size, 
                                        Qt.KeepAspectRatio, 
                                        Qt.SmoothTransformation)
            
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
        
        self.user_angle1 = 0
        self.user_angle2 = 0
        self.pro_angle1 = 0
        self.pro_angle2 = 0

        self.feedback_button = QPushButton("Get Feedback")
        self.feedback_button.clicked.connect(self.get_feedback)

        self.feedback_text = QLabel("")

        self.update_screenshot(self.screenshot_path)
        self.update_pro_photo(pro_player)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.user_dropdown)
        controls_layout.addWidget(self.pro_dropdown)
        
        layout.addWidget(controls_layout.addWidget(self.feedback_button))
        layout.addWidget(self.feedback_text)

        comparison_layout.addWidget(self.screenshot_label)
        comparison_layout.addWidget(self.comparison_photo_label)
        comparison_layout.addLayout(controls_layout)
        
        layout.addLayout(comparison_layout)
        self.setLayout(layout)

    def calculate_angle(self, first, mid, end):
        first = np.array(first)
        mid = np.array(mid)
        end = np.array(end)

        radians = np.arctan2(end[1] - mid[1], end[0] - mid[0]) - np.arctan2(first[1] - mid[1], first[0] - mid[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        
        return angle

    def draw_pose_on_image(self, image_path):
        image = cv2.imread(image_path)
        if image is not None:
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results_pose = self.pose.process(frame_rgb)

            if results_pose.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame_rgb, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            try:
                landmarks = results_pose.pose_landmarks.landmark
                
                self.right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                self.right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                self.right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                self.right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                self.left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                self.left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                self.average_x = (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x + landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x) / 2
                self.average_y = (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y + landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y) / 2

                self.crotch = [self.average_x, self.average_y]

                height, width = frame_rgb.shape[:2]
                self.right_hip = tuple(np.multiply(self.right_hip, [width, height]).astype(int))
                self.right_shoulder = tuple(np.multiply(self.right_shoulder, [width, height]).astype(int))
                self.right_elbow = tuple(np.multiply(self.right_elbow, [width, height]).astype(int))

                self.right_knee = tuple(np.multiply(self.right_knee, [width, height]).astype(int))
                self.left_hip = tuple(np.multiply(self.left_hip, [width, height]).astype(int))
                self.left_knee = tuple(np.multiply(self.left_knee, [width, height]).astype(int))
                self.crotch = tuple(np.multiply(self.crotch, [width, height]).astype(int))

                self.crotch_x = int(self.average_x * width)
                self.crotch_y = int(self.average_y * height)

                cv2.circle(frame_rgb, (self.crotch_x, self.crotch_y), radius = 5, color = (255, 0, 0), thickness = 10)

                angle1 = self.calculate_angle(self.right_hip, self.right_shoulder, self.right_elbow)
                angle2 = self.calculate_angle(self.right_knee, self.crotch, self.left_knee)

                if os.path.basename(image_path).startswith('contact_screenshot'):
                    self.user_angle1 = angle1
                    self.user_angle2 = angle2
                    print(f"User Arm & Body Angle: {self.user_angle1}, User Stance Angle: {self.user_angle2}")
                    print(f"User Arm & Body Angle: {self.user_angle1}, User Stance Angle: {self.user_angle2}")
                    cv2.putText(frame_rgb, str(self.user_angle1), self.right_shoulder, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame_rgb, str(self.user_angle2), self.crotch, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    self.pro_angle1 = angle1
                    self.pro_angle2 = angle2
                    print(f"Pro Arm & Body Angle: {self.pro_angle1}, Pro Stance Angle: {self.pro_angle2}")
                    cv2.putText(frame_rgb, str(self.pro_angle1), self.right_shoulder, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame_rgb, str(self.pro_angle2), self.crotch, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
            except:
                pass

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            pose_image_path = os.path.splitext(image_path)[0] + "_pose.png"
            cv2.imwrite(pose_image_path, frame_bgr)

            return pose_image_path

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
        updated_user_path = self.draw_pose_on_image(user_path)
        if user_path and os.path.exists(updated_user_path):
            user_photo = QPixmap(updated_user_path)
            self.screenshot_label.setPixmap(user_photo.scaled(400, 400, Qt.KeepAspectRatio))
        else:
            self.screenshot_label.clear()

    def update_pro_photo(self, player):
        pro_photos = {
            "Rafa": "/Users/yizhengc/dev/Quest_Project_Tennis/images/rafa_forehand_contact_ao.png",
            "Sinner": "/Users/yizhengc/dev/Quest_Project_Tennis/images/sinner_forehand_contact.png"
        }

        pro_path = pro_photos.get(player)
        updated_pro_path = self.draw_pose_on_image(pro_path)
        if pro_path and os.path.exists(updated_pro_path):
            comparison_photo = QPixmap(updated_pro_path)
            self.comparison_photo_label.setPixmap(comparison_photo.scaled(400, 400, Qt.KeepAspectRatio))
        else:
            self.comparison_photo_label.clear()

    def get_feedback(self):
        feedback_text = "Feedback: "
        
        arm_angle_diff = abs(self.user_angle1 - self.pro_angle1)
        if arm_angle_diff > 15:
            if self.user_angle1 < self.pro_angle1:
                feedback_text += "Arm is too closed. "
            else:
                feedback_text += "Arm is over-extended. "
        else:
            feedback_text += "Good arm position. "
        
        stance_angle_diff = abs(self.user_angle2 - self.pro_angle2)
        if stance_angle_diff > 15:
            if self.user_angle2 < self.pro_angle2:
                feedback_text += "Stance is too narrow. "
            else:
                feedback_text += "Stance is too wide. "
        else:
            feedback_text += "Good stance. "
        
        self.feedback_text.setText(feedback_text)

class VideoPlayer(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.contact_detected = False 
        self.screenshot_path = None
        self.parent = parent

        self.setWindowTitle("Video Player with Frame Modification")
        self.setGeometry(200, 100, 800, 600)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoSurface = CustomVideoSurface(self)
        self.mediaPlayer.setVideoOutput(self.videoSurface)

        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setMinimumSize(640, 480)
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoSurface.setWidget(self.videoLabel)

        self.openButton = QPushButton("Open Video")
        self.openButton.clicked.connect(self.open_file)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play_video)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)

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
        self.selected_player = "Rafa"

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