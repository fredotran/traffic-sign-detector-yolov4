import os
import zipfile
import shutil
import wget
import glob


##################################################################
####################### DOWNLOAD DATA ############################
##################################################################

downloaded_data_dir = 'downloaded_sets'
parent_dir = '../traffic-signs-detection/'
downloaded_data_path = os.path.join(parent_dir, downloaded_data_dir)

if not os.path.exists(downloaded_data_dir):
    os.mkdir(downloaded_data_path)
else: 
    pass

# download the last version of YOLOv4 darknet
url = "https://github.com/fredotran/traffic-signs-detection/releases/download/weights/Traffic.Road.Signs.YoloV3.format.v2-10-01-2021.darknet.zip"
wget.download(url, downloaded_data_path)

##################################################################
###################### DATA EXTRACTION ###########################
##################################################################

print("\n Start extracting!")

# extract the downloaded files in the main directory
for zipfiles in glob.iglob(os.path.join(downloaded_data_path, "*.zip")):
    with zipfile.ZipFile(zipfiles, 'r') as zipObj:
       # Extract all the contents of zip file in the parent directory      
       zipObj.extractall(parent_dir)

print("\n Extraction complete!")
