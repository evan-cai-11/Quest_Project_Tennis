import torch
import cv2

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt')

# Path to the tennis video (replace with your actual video path)
video_path = '/Users/yizhengc/Downloads/QPV/Sinner.mp4'  # Change this to your actual video file path

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Process video frame by frame
while True:
    ret, frame = cap.read()

    # Break the loop if the video ends
    if not ret:
        break

    # Convert the frame (image) to a format that YOLOv7 can process
    results = model(frame)

    # Draw bounding boxes on the frame for detected objects
    for det in results.xyxy[0]:  # For each detected object
        xmin, ymin, xmax, ymax, conf, cls = det.tolist()

        # Filter out low-confidence detections (confidence > 0.5)
        if conf > 0.5:
            # Draw a rectangle (bounding box) around the object
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            # Get the class label
            label = f'{model.names[int(cls)]}: {conf:.2f}'
            # Put the label text above the bounding box
            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame (optional, comment this out if not needed)
    cv2.imshow('YOLOv7 Tennis Posture Detection', frame)

    # Press 'q' to exit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()

# Close any OpenCV windows
cv2.destroyAllWindows()