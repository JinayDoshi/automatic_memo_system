# import cv2
# import numpy as np
# import torch
# from datetime import datetime
# import pytesseract
# from ultralytics import YOLO  # Assuming YOLOv8

# # Load models
# helmet_detector = YOLO('helmet-detector.pt')
# vehicle_detector = YOLO('yolov8s.pt')
# number_plate_detector = YOLO('license_plate_detector.pt')

# # Initialize tracking dictionary for unique IDs
# unique_id_counter = 0
# tracked_objects = {}

# # Annotator line position (25% from the bottom)
# def get_annotator_line(frame_height):
#     return int(frame_height * 0.75)

# # Function to perform OCR
# pytesseract.pytesseract.tesseract_cmd = r'"C:/Program Files/Tesseract-OCR/tesseract.exe"'  # Update path if necessary
# def perform_ocr(cropped_image):
#     gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     text = pytesseract.image_to_string(gray, config='--psm 8')
#     return text.strip()

# # Function to generate a unique ID
# def generate_unique_id():
#     global unique_id_counter
#     unique_id_counter += 1
#     return f"ID_{unique_id_counter}"

# # Function to update and store detected object data
# def store_data(unique_id, vehicle_type, helmet_status, number_plate):
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     date, time = current_time.split(' ')
#     data_entry = f"{unique_id},{date},{time},{vehicle_type},{helmet_status},{number_plate}"
#     print(data_entry)  # You can write this to a file or database

# # Process video
# cap = cv2.VideoCapture('C:/Users/JINAY DOSHI/Desktop/auto memo generator/testvideos/test3.mp4')  # Update with your video source
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# annotator_line = get_annotator_line(frame_height)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect vehicles and helmets
#     vehicle_results = vehicle_detector(frame)
#     helmet_results = helmet_detector(frame)
    
#     # Ensure results are processed correctly
#     if len(vehicle_results):
#         vehicle_boxes = vehicle_results[0].boxes  # Access the first result (assumes one frame at a time)
#         if vehicle_boxes is not None:
#             for box in vehicle_boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()[:4]
#                 confidence = box.conf[0].item()
#                 class_id = int(box.cls[0].item())
#                 vehicle_type = vehicle_detector.names[class_id]

#                 # Track new object and assign unique ID if not tracked
#                 unique_id = tracked_objects.get((x1, y1, x2, y2), generate_unique_id())
#                 tracked_objects[(x1, y1, x2, y2)] = unique_id

#                 # Detect helmet status for motorcycles
#                 helmet_status = "NONE"
#                 if vehicle_type == "motorcycle" and len(helmet_results):
#                     helmet_boxes = helmet_results[0].boxes
#                     if helmet_boxes is not None:
#                         for h_box in helmet_boxes:
#                             helmet_status = helmet_detector.names[int(h_box.cls[0].item())]

#                 # Draw bounding box
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{unique_id} {vehicle_type} {helmet_status}", (int(x1), int(y1) - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#                 # Check if vehicle crosses annotator line
#                 if y2 >= annotator_line:
#                     number_plate_results = number_plate_detector(frame[int(y1):int(y2), int(x1):int(x2)])
#                     number_plate_text = "UNKNOWN"
#                     if len(number_plate_results):
#                         np_boxes = number_plate_results[0].boxes
#                         if np_boxes is not None:
#                             for np_box in np_boxes:
#                                 np_x1, np_y1, np_x2, np_y2 = np_box.xyxy[0].tolist()[:4]
#                                 cropped_np = frame[int(np_y1):int(np_y2), int(np_x1):int(np_x2)]
#                                 number_plate_text = perform_ocr(cropped_np)

#                     # Store data
#                     store_data(unique_id, vehicle_type, helmet_status, number_plate_text)



#     # Draw annotator line (invisible)
#     cv2.line(frame, (0, annotator_line), (frame_width, annotator_line), (0, 0, 0), 1)

#     # Display frame
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# def store_data(unique_id, vehicle_type, helmet_status, number_plate):
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     date, time = current_time.split(' ')
#     data_entry = f"{unique_id},{date},{time},{vehicle_type},{helmet_status},{number_plate}"
    
#     # Print to console
#     print(data_entry)
    
#     # Save to file
#     with open("detected_objects.csv", "a") as file:
#         file.write(data_entry + "\n")

# cap.release()
# cv2.destroyAllWindows()


import cv2
import torch
import pytesseract
import numpy as np
from datetime import datetime

# Load the YOLOv8 model, helmet detector model, and number plate detector model
vehicle_model = torch.hub.load('ultralytics/yolov8', 'yolov8s.pt')  # YOLOv8 model
helmet_model = torch.hub.load('ultralytics/yolov8', 'helmet-detector.pt')  # Helmet detector model
license_plate_model = torch.hub.load('ultralytics/yolov8', 'license_plate_detector.pt')  # License plate detector model

# Initialize unique ID counter and dictionary for tracking vehicles and drivers
unique_id_counter = 0
tracked_objects = {}

# Function to perform OCR on detected number plates
def ocr_number_plate(image, bbox):
    # Crop the detected number plate region
    x1, y1, x2, y2 = bbox
    number_plate_img = image[y1:y2, x1:x2]
    
    # Perform OCR on the number plate
    ocr_result = pytesseract.image_to_string(number_plate_img, config='--psm 8').strip()
    return ocr_result

# Function to detect and process vehicles and drivers
def process_frame(frame):
    global unique_id_counter, annotator_line
    
    # Get the current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Get the frame height and width
    frame_height, frame_width, _ = frame.shape
    
    # Invisible annotator line at 25% from the bottom
    annotator_line = int(frame_height * 0.75)
    
    # Detect vehicles using YOLOv8 model
    results = vehicle_model(frame)
    vehicle_detections = results.xywh[0].numpy()

    # Detect drivers using helmet detector
    driver_results = helmet_model(frame)
    driver_detections = driver_results.xywh[0].numpy()

    # Loop through all detected vehicles
    for det in vehicle_detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) in [2, 5, 7]:  # 2: car, 5: bus, 7: truck (YOLOv8 class labels)
            # Draw bounding box for detected vehicle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"Vehicle {unique_id_counter}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Assign a unique ID to this vehicle if it's not already assigned
            vehicle_id = f"vehicle_{unique_id_counter}"
            if vehicle_id not in tracked_objects:
                tracked_objects[vehicle_id] = {
                    'type': 'vehicle',
                    'unique_id': vehicle_id,
                    'bbox': (x1, y1, x2, y2),
                    'timestamp': current_time,
                    'ocr_done': False
                }
                unique_id_counter += 1

    # Loop through all detected drivers
    for det in driver_detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) in [1, 2]:  # 1: driver, 2: bicyclist (YOLOv8 class labels)
            # Draw bounding box for detected driver
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Driver {unique_id_counter}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Perform helmet detection
            helmet_status = 'no-helmet'  # Default assumption
            if conf > 0.5:
                helmet_status = 'helmet' if helmet_model(frame)[0][0][0] == 1 else 'no-helmet'
            
            # Assign a unique ID to this driver if not already assigned
            driver_id = f"driver_{unique_id_counter}"
            if driver_id not in tracked_objects:
                tracked_objects[driver_id] = {
                    'type': 'driver',
                    'unique_id': driver_id,
                    'bbox': (x1, y1, x2, y2),
                    'helmet_status': helmet_status,
                    'timestamp': current_time
                }
                unique_id_counter += 1

    # Detect number plates
    plate_results = license_plate_model(frame)
    plate_detections = plate_results.xywh[0].numpy()

    # Process detected number plates and perform OCR
    for det in plate_detections:
        x1, y1, x2, y2, conf, cls = det
        number_plate = ocr_number_plate(frame, (x1, y1, x2, y2))

        # Draw bounding box for detected number plate
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, f"Plate {number_plate}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Check if the number plate touches the annotator line (25% from bottom)
        if y2 >= annotator_line and not tracked_objects.get(f"plate_{det[0]}", {}).get('ocr_done', False):
            vehicle_data = tracked_objects.get(f"vehicle_{det[0]}")
            if vehicle_data:
                # Save data to the database or file (this is just an example of printing)
                print(f"Data Saved: {vehicle_data['unique_id']}, {vehicle_data['timestamp']}, "
                      f"Vehicle Type: {vehicle_data['type']}, Helmet Status: {vehicle_data.get('helmet_status', 'N/A')}, "
                      f"Number Plate: {number_plate}")
                vehicle_data['ocr_done'] = True

    return frame

# Initialize video capture
cap = cv2.VideoCapture('C:/Users/JINAY DOSHI/Desktop/auto memo generator/testvideos/test3.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow('Processed Frame', processed_frame)

    # Break on ESC key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
