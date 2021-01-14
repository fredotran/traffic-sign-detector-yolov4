import os
import shutil
import wget

directory = 'weights'
parent_dir = 'C:/Users/taket/Desktop/Jupyter/traffic-signs-detection/'
path = os.path.join(parent_dir, directory)

# if there is no 'weights' directory, makes one
if not os.path.exists(directory):
    true_path = os.mkdir(path)
else:
    pass

print("\n Start downloading YoloV4 weights...!")

# download the last version of YOLOv4 darknet from Alexey's repository
url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
wget.download(url, path)

print("\n Downloading complete!\n")

print("\n Start downloading YoloV4 custom RDS version weights...!")

# download the my custom version of YOLOv4 for traffic road signs detection
url = "https://github.com/fredotran/traffic-signs-detection/releases/download/weights/yolov4-rds_best_2000.weights"
wget.download(url, path)

print("\n Downloading complete!")