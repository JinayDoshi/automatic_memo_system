import cv2
import csv
import datetime
from ultralytics import YOLO

# Load the custom-trained YOLO model
helmet_model = YOLO('helmet-detector.pt')  # Custom model for drivers and helmets

# Video input
video_path = 'C:/Users/JINAY DOSHI/Desktop/auto memo generator/testvideos/test3.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
output_video_path = 'C:/Users/JINAY DOSHI/Desktop/auto memo generator/outputvideo/output_with_helmet_status.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Dictionary to store unique IDs for detected drivers
driver_ids = {}
next_id = 1

# Open the CSV file for saving results
csv_file = 'helmet_detection_results.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Unique_ID', 'Date', 'Time', 'Helmet_Status'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform helmet detection on the frame
        results = helmet_model(frame)

        # Loop through each detected object
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract the bounding box coordinates and class label
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Class labels: 0 = driver, 1 = helmet, 2 = no_helmet
                    if class_id == 0:  # Driver detected
                        # Assign a unique ID if not already assigned
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        driver_key = (center_x, center_y)

                        if driver_key not in driver_ids:
                            driver_ids[driver_key] = next_id
                            unique_id = next_id
                            next_id += 1
                        else:
                            unique_id = driver_ids[driver_key]

                        # Check if the driver is wearing a helmet or not
                        helmet_status = 'Unknown'
                        for box2 in boxes:
                            class_id2 = int(box2.cls[0])
                            if class_id2 == 1:  # Helmet detected
                                helmet_status = 'Helmet'
                            elif class_id2 == 2:  # No helmet detected
                                helmet_status = 'No Helmet'

                        # Draw the bounding box and label
                        color = (0, 255, 0) if helmet_status == 'Helmet' else (0, 0, 255)
                        label = f'ID {unique_id} - {helmet_status}'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Get the current date and time
                        now = datetime.datetime.now()
                        date_str = now.strftime('%Y-%m-%d')
                        time_str = now.strftime('%H:%M:%S')

                        # Write the result to the CSV file
                        writer.writerow([unique_id, date_str, time_str, helmet_status])

        # Write the frame to the output video
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Helmet Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to {output_video_path}")
print(f"Results saved to {csv_file}")
