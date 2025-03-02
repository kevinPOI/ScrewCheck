from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import cv2
import os

# Load YOLO model using SAHI wrapper
# detection_model = Yolov8DetectionModel(
#     model_path="allen-head-small.pt",
#     confidence_threshold=0.1,  # Adjust confidence threshold
#     iou_threshold=0.3,         # Adjust IoU threshold
#     device="cuda",             # Use GPU if available
# )
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolo11",
    model_path="allen-head-small-v2.pt",
    confidence_threshold=0.1,
    device="cuda:0",  # or 'cuda:0'
)
# Define input and output folders
input_folder = "test_imgsHD"
output_folder = "test_results"
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png')):
        file_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load image
        image = cv2.imread(file_path)
        height, width, _ = image.shape

        # Compute crop boundaries (middle 70% in x-axis)
        crop_x_start = int(0.15 * width)
        crop_x_end = int(0.85 * width)

        # Crop the image
        cropped_image = image[:, crop_x_start:crop_x_end]

        # Perform sliced detection
        result = get_sliced_prediction(
            cropped_image,
            detection_model,
            slice_height=320, 
            slice_width=320,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1
        )

        # Save detection result image
        result.export_visuals(export_dir=output_folder, file_name=filename)

        # Count detected objects
        num_screws = len(result.object_prediction_list)

        # Print pass/fail based on number of screws
        if num_screws == 3:
            print(f"PASS ({num_screws} screws detected) - {filename}")
        else:
            print(f"FAIL ({num_screws} screw(s) detected) - {filename}")

print(f"\nSliced detection complete! Results saved in '{output_folder}'")
