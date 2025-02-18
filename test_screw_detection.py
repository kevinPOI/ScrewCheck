from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import os
model = YOLO("allen-head-hd.pt")

input_folder = "test_imgs2"
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    
    # Check if the file is an image
    if filename.lower().endswith(('.jpg', '.png')):
        # Load image with OpenCV
        image = cv2.imread(file_path)
        # Run inference
        height, width, _ = image.shape

# Compute crop boundaries (middle 70% in x-axis)
        crop_x_start = int(0.15 * width)
        crop_x_end = int(0.85 * width)

        # Crop the image
        image = image[:, crop_x_start:crop_x_end]
        results = model(image, imgsz = 1280, show = True, conf = 0.1, iou = 0.3, device = 'cuda')
        boxes = results[0].boxes
        num_screws = boxes.xywh.shape[0]
        if num_screws == 3:
            print("PASS (", num_screws, "screws detected)")
        else:
            print("FAIL (", num_screws, "screw(s) detected)")
        input("Press Enter to continue...")