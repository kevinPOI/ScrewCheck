from roboflow import Roboflow
from ultralytics import YOLO
# rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
# project = rf.workspace("test-5ev0m").project("mid_of_studs_v2")
# version = project.version(3)
# dataset = version.download("yolov8")
                

# model = YOLO("yolov8n-pose.pt")  # load a pretrained model. change this to the model you want to use, n, s, l etc. (only ending)

# # Train the model


# rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
# project = rf.workspace("test-5ev0m").project("allen-head-screw-detection")
# version = project.version(1)
# dataset = version.download("yolov11")


from roboflow import Roboflow
rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
project = rf.workspace("test-5ev0m").project("allen-head-screw-detection")
version = project.version(6)
dataset = version.download("yolov11")
model = YOLO("yolo11s.pt")  # load a pretrained model. change this to the model you want to use, n, s, l etc. (only ending)
save_dir = ("runs_locate")
results = model.train(data="allen-head-screw-detection-6/data.yaml", imgsz=1280, batch=6, epochs=50, plots=True, save_dir = save_dir)   

# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
# project = rf.workspace("test-5ev0m").project("allen-head-screw-detection")
# version = project.version(3)
# dataset = version.download("yolov11")
                