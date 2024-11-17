from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtMultimedia import QMediaPlayer, QVideoSink
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

import sys

class VideoProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.media_player = QMediaPlayer(self)
        
        # QVideoSink to capture frames
        self.video_sink = QVideoSink(self)
        self.media_player.setVideoOutput(self.video_sink)
        
        # Connect to capture each frame
        self.video_sink.videoFrameChanged.connect(self.process_frame)

        # Display for modified frame
        self.modified_frame_label = QLabel(self)
        self.modified_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.modified_frame_label)
        self.setLayout(layout)

    def play_video(self, video_url):
        self.media_player.setSource(QUrl.fromLocalFile(video_url))
        self.media_player.play()

    def process_frame(self, frame):
        # Convert QVideoFrame to QImage
        if frame.isValid():
            image = frame.toImage()
            if image.isNull():
                return

            # Modify the image (example: invert colors)
            modified_image = image.convertToFormat(QImage.Format.Format_RGB32)
            for y in range(modified_image.height()):
                for x in range(modified_image.width()):
                    color = modified_image.pixel(x, y)
                    inverted_color = QColor(255 - QColor(color).red(),
                                            255 - QColor(color).green(),
                                            255 - QColor(color).blue())
                    modified_image.setPixel(x, y, inverted_color.rgb())

            # Convert modified QImage to QPixmap and display it
            pixmap = QPixmap.fromImage(modified_image)
            self.modified_frame_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    processor = VideoProcessor()
    processor.play_video("/Users/yizhengc/Downloads/Sinner2.mp4")  # Replace with your video path
    processor.show()
    sys.exit(app.exec())