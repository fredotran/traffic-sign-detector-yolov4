import os
import shutil
import glob
from shutil import copyfile

print("\n Start Processing!")

# parent directory (change parent directory for your own)
parent_dir = "../traffic-signs-detection/"

# training, validation and test sets folders (arbitrary paths)
train_dir = parent_dir + "train" 
validation_dir = parent_dir + "valid" 
test_dir = parent_dir + "test" 
data_dir = parent_dir + "data"
weights_dir = parent_dir + "weights"

##############################################################
################ COPY CLASSNAMES IN OBJ.NAMES ################
##############################################################

# create data folder
data_path = os.path.join(parent_dir, data_dir)

if not os.path.exists(data_path):
    os.mkdir(data_path)
else:
    pass

# go to copy the labels into obj.names files
src = train_dir + "/_darknet.labels"
dst = data_dir + "/obj.names" 
shutil.copyfile(src, dst)

# create folder "obj"
directory = "obj"
path_obj = os.path.join(data_dir, directory)

if not os.path.exists(path_obj):
    os.mkdir(path_obj)
else:
    pass

##############################################################
###################### TRAINING DATA #########################
##############################################################

# get the training images then copy them to the obj folder
train_imgs_src = train_dir
train_imgs_dst = data_dir + "/obj"

# create a "train" folder into "obj" folder
obj_train_dir = "train_processed"
path_train_dst = os.path.join(train_imgs_dst, obj_train_dir)
os.mkdir(path_train_dst)

# copy images from the source to the "train folder" in "obj" folder
for jpgfile in glob.iglob(os.path.join(train_imgs_src, "*.jpg")):
    shutil.copy(jpgfile, path_train_dst)
    
##############################################################
#################### VALIDATION DATA #########################
##############################################################

# get the validation images then copy them to the obj folder  
valid_imgs_src = validation_dir
valid_imgs_dst = data_dir + "/obj"

# create a "valid" folder into "obj" folder
obj_val_dir = "valid_processed"
path_valid_dst = os.path.join(valid_imgs_dst, obj_val_dir)
os.mkdir(path_valid_dst)

# copy images from the source to the "valid folder" in "obj" folder
for jpgfile in glob.iglob(os.path.join(valid_imgs_src, "*.jpg")):
    shutil.copy(jpgfile, path_valid_dst)    
    
##############################################################
####################### TEST DATA ############################
##############################################################

# get the validation images then copy them to the obj folder  
test_imgs_src = test_dir
test_imgs_dst = data_dir + "/obj"

# create a "valid" folder into "obj" folder
obj_test_dir = "test_processed"
path_test_dst = os.path.join(test_imgs_dst, obj_test_dir)
os.mkdir(path_test_dst)

# copy images from the source to the "valid folder" in "obj" folder
for jpgfile in glob.iglob(os.path.join(test_imgs_src, "*.jpg")):
    shutil.copy(jpgfile, path_test_dst)  
    
    
##############################################################
###################### LABELS DATA ###########################
##############################################################

# copy training images' labels from the source to the "train folder" in "obj" folder
for labels in glob.iglob(os.path.join(train_imgs_src, "*.txt")):
    shutil.copy(labels, path_train_dst)

# copy validation images' labels from the source to the "valid folder" in "obj" folder
for labels in glob.iglob(os.path.join(valid_imgs_src, "*.txt")):
    shutil.copy(labels, path_valid_dst)    
    
# copy test images' labels from the source to the "test folder" in "obj" folder
for labels in glob.iglob(os.path.join(test_imgs_src, "*.txt")):
    shutil.copy(labels, path_test_dst)    

##############################################################
##############################################################

# creating backup folder
backup_folder = "backup"
backup_dir = os.path.join(parent_dir, backup_folder)

if not os.path.exists(backup_dir):
    os.mkdir(backup_dir)
else:
    pass
    
##############################################################
############### WRITTING FILENAMES IN OBJ.DATA ###############
##############################################################

# writting filenames in the obj.data file 
with open((parent_dir+'data/obj.data'), 'w') as out:
    out.write('classes = 4\n')
    out.write('train = ../data/train.txt\n')
    out.write('valid = ../data/valid.txt\n')
    out.write('test = ../data/test.txt\n')
    out.write('names = ../data/obj.names\n')
    out.write('backup = ../backup/')

##############################################################
############## WRITTING IMG NAMES IN TXT FILES ###############
##############################################################

# write the train.txt file with a list of the training images stored in /data/obj/train
with open('../traffic-signs-detection/data/train.txt', 'w') as out:
    for img in [f for f in os.listdir(train_dir) if f.endswith('jpg')]:
        out.write(('../data/obj/train_processed/') + img + '\n')

# write the valid.txt file with a list of the validation images stored in /data/obj/valid
with open('../traffic-signs-detection/data/valid.txt', 'w') as out:
    for img in [f for f in os.listdir(validation_dir) if f.endswith('jpg')]:
        out.write(('../data/obj/valid_processed/') + img + '\n')

# write the valid.txt file with a list of the validation images stored in /data/obj/valid
with open('../traffic-signs-detection/data/test.txt', 'w') as out:
    for img in [f for f in os.listdir(test_dir) if f.endswith('jpg')]:
        out.write(('../data/obj/test_processed/') + img + '\n')
        
print("\n Processing complete!")
