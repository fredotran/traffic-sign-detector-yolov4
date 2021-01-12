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
url = "https://app.roboflow.com/ds/sXWjsufRH3?key=xroPHpt1sy"
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
















