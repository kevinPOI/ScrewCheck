from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch


# Load YOLO model

# NMS helper function to remove overlapping detections within 10 pixels
def apply_nms(boxes, threshold=100):
    filtered_boxes = []
    for box in boxes:
        x1, y1, x2, y2, conf = box
        keep = True
        for fbox in filtered_boxes:
            fx1, fy1, fx2, fy2, _ = fbox
            # Check if the box is within the threshold distance
            if abs(x1 - fx1) < threshold and abs(y1 - fy1) < threshold:
                keep = False
                break
        if keep:
            filtered_boxes.append(box)
    return filtered_boxes

# Process each image in the input folder
def run_test():
    model = YOLO("allen-head-small-v3.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  
    # Define input folder
    input_folder = "test_imgs_HD_sample"
    debug = False
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            file_path = os.path.join(input_folder, filename)
            
            # Load image
            image = cv2.imread(file_path)
            height, width, _ = image.shape

            # Compute crop boundaries (middle 70% in x-axis)
            crop_x_start = int(0.15 * width)
            crop_x_end = int(0.85 * width)

            # Crop the image
            cropped_image = image[:, crop_x_start:crop_x_end]

            # Resize the cropped image to 1280x1280
            resized_image = cv2.resize(cropped_image, (1280, 1280))

            # Define patch size before resizing
            patch_original_size = 380
            patch_resized_size = 300
            stride = 320  # Overlapping patches

            num_screws_total = 0
            detected_boxes = []

            print(f"\nProcessing {filename}...")

            # Slice the image into 16 overlapping patches (4x4 grid)
            for row in range(4):
                for col in range(4):
                    x_start = col * stride
                    y_start = row * stride
                    x_end = x_start + patch_original_size
                    y_end = y_start + patch_original_size

                    # Extract patch
                    patch = resized_image[y_start:y_end, x_start:x_end]

                    # Resize the patch to 320x320
                    patch_resized = cv2.resize(patch, (patch_resized_size, patch_resized_size))

                    # Run YOLO inference on the patch
                    results = model(patch_resized, imgsz=320, conf=0.5, iou=0.3, device=device, show = debug)
                    
                    boxes = results[0].boxes
                    if debug and len(boxes) > 0:
                        input()
                    # Store detected bounding boxes
                    for box in boxes:
                        x_center, y_center, w, h= box.xywhn[0].cpu().numpy()  # Normalized values
                        conf = box.conf
                        # Convert to original patch scale (360x360)
                        x_center *= patch_original_size
                        y_center *= patch_original_size
                        w *= patch_original_size
                        h *= patch_original_size

                        # Convert local (patch) coordinates to global (full image) coordinates
                        abs_x_center = x_center + x_start
                        abs_y_center = y_center + y_start
                        abs_x_min = int(abs_x_center - w / 2)
                        abs_y_min = int(abs_y_center - h / 2)
                        abs_x_max = int(abs_x_center + w / 2)
                        abs_y_max = int(abs_y_center + h / 2)

                        detected_boxes.append([abs_x_min, abs_y_min, abs_x_max, abs_y_max, float(conf)])

            # Apply NMS to remove overlapping detections
            filtered_boxes = apply_nms(detected_boxes)

            # Draw final bounding boxes on the full image
            for box in filtered_boxes:
                x1, y1, x2, y2, conf = box
                num_screws_total += 1
                cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(resized_image, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print total detections for the full image
            print(f"Total screws detected in {filename} (after NMS): {num_screws_total}")

            # Show the final image with bounding boxes
            cv2.imshow("Detection Results", resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test()