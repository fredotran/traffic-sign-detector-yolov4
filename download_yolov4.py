import os
import shutil
import wget

print("\n Start downloading weights...!")

directory = 'weights'
parent_dir = 'C:/Users/taket/Desktop/Jupyter/traffic-signs-detection/'
path = os.path.join(parent_dir, directory)

# if there is no 'weights' directory, makes one
if not os.path.exists(directory):
    true_path = os.mkdir(path)
else:
    pass

# download the last version of YOLOv4 darknet
url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
wget.download(url, path)

print("\n Downloading complete!")