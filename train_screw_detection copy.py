from roboflow import Roboflow
from ultralytics import YOLO

mode = 3

if mode == 0:
    from roboflow import Roboflow
    rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
    project = rf.workspace("test-5ev0m").project("allen-head-screw-detection")
    version = project.version(6)
    dataset = version.download("coco")

if mode == 1:
    from sahi.slicing import slice_coco

    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path="allen-head-screw-detection-6/valid/_annotations.coco.json",
        image_dir="allen-head-screw-detection-6/valid",
        output_coco_annotation_file_name= "_annotations.json_coco.json",
        output_dir="valid",
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
if mode == 2:
    from ultralytics.data.converter import convert_coco

    convert_coco("valid", "valid_yolo", use_segments=False, use_keypoints=False, cls91to80=False)

if mode == 3:
    model = YOLO("yolo11s.pt")  # load a pretrained model. change this to the model you want to use, n, s, l etc. (only ending)
    save_dir = ("runs_locate")
    results = model.train(data="allen-head-screw-detection-6/data.yaml", imgsz=256, batch=64, epochs=50, plots=True, save_dir = save_dir)   

# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
# project = rf.workspace("test-5ev0m").project("allen-head-screw-detection")
# version = project.version(3)
# dataset = version.download("yolov11")
                