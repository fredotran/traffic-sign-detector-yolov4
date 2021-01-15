# import useful libraries
import os
import numpy as np
import cv2

# function to load our classes names
def read_classes(file):
    """ Read the classes files and extract the classes' names in it""" 
    classNames = []    
    with open(file, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
   
    return classNames


# object detection function
def object_detection(outputs, input_frame, confidenceThreshold):
    """ This function will perform bounding boxes acquisition on detected objects in frames (images) """
    # first we'll collect the height, width and channel of the input frame (3 if the image is RGB, 1 if it's grayscale)
    height, width, channel = input_frame.shape
    
    # we'll create category lists to store the layers' output values 
    bounding_boxes = []
    class_objects = []
    confidence_probs = []
    
    # Knowing that there are 3 YOLO layers, we'll browsing them and their outputs using this :
    for result in outputs:        
        for values in result:
            
            scores = values[5:] # we know that the class probabilities are from the 5th values
            indices_object = np.argmax(scores) # get the indice of the max score
            confidence_probability = scores[indices_object] # store the maximum value of the indice found
            
            # in order to have a proper detection, we'll eliminate the weakest probability detection by imposing a threshold
            if confidence_probability > confidenceThreshold:   
                
                # get the pixel values corresponding to the scaling of the bounding box coordinates to the initial frame
                box_detected = values[0:4] * np.array([width, height, width, height])                                 
                # get the top left corner coordinates by extracting values from box_detected and perform calculations
                x, y, w, h = box_detected
                # we're converting the coordinates to int because OpenCV doesn't allow floats for bounding boxes
                x = int(x - (w/2))
                y = int(y - (h/2))
                
                # adding the good detected boxe in the bounding boxes list created
                bounding_boxes.append([x,y,w,h])                
                # adding the detected objects indices in the class objects list 
                class_objects.append(indices_object)                
                # adding the max value of the object score (confidence) in the confidences_probs list
                confidence_probs.append(float(confidence_probability))    
    
    return bounding_boxes, class_objects, confidence_probs



# non max-suppression function
def nms_bbox(bounding_boxes, confidence_probs, confidenceThreshold, nmsThreshold):
    """This function performs non-max suppression on all the bounding boxes detected and keeps the best one"""    
    #Using OpenCV DNN non-max supression to get the best bounding box of the detected object (retrieve the indices)
    indices_bbox = cv2.dnn.NMSBoxes(bounding_boxes, confidence_probs, confidenceThreshold, nmsThreshold)   
    print('Number of objects detected : ', len(indices_bbox), '\n')
    
    return indices_bbox


# box drawing functions
def box_drawing(input_frame, indices, bounding_boxes, class_objects, confidence_probs, classNames, color=(0,255,255), thickness=2):
    """ Drawing the detected objects boxes """
    # once we have the indices, we'll extract the values of x,y,w,h of the best bounding boxes and store them.
    for i in indices:
        i = i[0]
        final_box = bounding_boxes[i]
    # we'll retrieve the bounding boxes values (coordinates) now and use them to draw our boxes.
        x, y, w, h = final_box[0], final_box[1], final_box[2], final_box[3]
        x, y, w, h = int(x), int(y), int(w), int(h)        
        print('Bounding box coordinates in the frame : ', 'x : ', x,'|| y : ',y,'|| w : ',w,'|| h :',h , '\n')
    
        cv2.rectangle(input_frame, (x,y), (x+w,y+h),  color, thickness)
        cv2.putText(input_frame, f'{classNames[class_objects[i]].upper()} {int(confidence_probs[i]*100)}%',
                        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  color, thickness=1)
    

# converting input frame into blob format for OpenCV DNN
def convert_to_blob(input_frame, network, height, width):
    """ This function allow us to convert a frame/image into blob format for OpenCV DNN"""    
    blob = cv2.dnn.blobFromImage(input_frame, 1/255, (height,width), [0,0,0], 1, crop=False)
    network.setInput(blob)
    # get the YOLO output layers numbers (names), these layers will be useful for the detection part
    # the layer's name : yolo_82, yolo_94, yolo_106
    yoloLayers = network.getLayerNames()
    outputLayers = [(yoloLayers[i[0]-1]) for i in network.getUnconnectedOutLayers()]
    # Doing forward propagation with OpenCV
    outputs = network.forward(outputLayers)
    
    return outputs


# method to load our image from directory
def load_image(image_path):    
    """ Loading the image with OpenCV by inputing the path /[Classes]/[ImageName]""" 
    # get the image path and load the image
    img_full_path = './inputs/images' + image_path
    image = cv2.imread(img_full_path)
    
    return image


# method to load our video from directory
def load_video(video_path):    
    """ Loading the video with OpenCV by inputing the path /[Classes]/[VideoName]""" 
    # get the video path and load the video
    video_full_path = './inputs/videos' + video_path
    # load the video
    cap_video = cv2.VideoCapture(video_full_path)
    
    return cap_video


