import torch
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

# Path to the tennis video (replace with your actual video path)
video_path = '/Users/yizhengc/PycharmProjects/TennisCoach/Sinner.mp4'

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection using YOLOv7
    results = model(frame)

    # Loop through detected objects and draw bounding boxes
    for det in results.xyxy[0]:  # Results for the first image
        xmin, ymin, xmax, ymax, conf, cls = det.tolist()

        # Filter out low-confidence detections (confidence > 0.5)
        if conf > 0.5:
            # Draw bounding box
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

            # Display class label (change if necessary)
            label = f'{model.names[int(cls)]}: {conf:.2f}'
            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    poses = pose.process(image_rgb)

    # Draw the pose on the image
    if poses.pose_landmarks:
        mp_drawing.draw_landmarks(frame, poses.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Tennis Racket Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
