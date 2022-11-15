# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

from flask import Flask, render_template, request, session, Response
from camera import VideoCamera
import pandas as pd
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.io.json import json_normalize
import csv
import base64
import json
import pickle
from werkzeug.utils import secure_filename
 
#*** Backend operation
 
# WSGI Application
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name
 
# Accepted image for to upload for object detection model
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'You Will Never Guess'



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#image = cv2.imread(uploaded_image_path)

def detect_object(uploaded_image_path):
    # Loading image
    image = cv2.imread(uploaded_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.5,
        skip_scores=True,
        skip_labels=True)

    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)
    # Write output image (object detection output)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image_detection.jpg')
    cv2.imwrite(output_image_path, image)
    return(output_image_path)

def obscure_object(uploaded_image_path):
    # Loading image
    image = cv2.imread(uploaded_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=-1,
        min_score_thresh=1,
        skip_scores=True,
        skip_labels=True)

    # All the results have been drawn on image. Now display the image.
    true_boxes = boxes[0][scores[0] > 0.75]
    if np.any(true_boxes):
        height, width, channels = image.shape
        ymin = true_boxes[0,0]*height
        xmin = true_boxes[0,1]*width
        ymax = true_boxes[0,2]*height
        xmax = true_boxes[0,3]*width
        img_gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        B1=image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),2].shape
        im=image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),:]
        #img_gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        thresh = threshold_otsu(img_gray)
        img_otsu  = img_gray < thresh
        print(img_gray*img_otsu) 
        #image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),0]=np.random.randint(50, size=(B1[0],B1[1]))
        #image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),1]=np.random.randint(50, size=(B1[0],B1[1]))
        #image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),2]=np.random.randint(10, size=(B1[0],B1[1]))
        image[:,:,0]=img_gray*img_otsu
        image[:,:,1]=img_gray*img_otsu
        image[:,:,2]=img_gray*img_otsu
    cv2.imshow('User Interface detector', image)
    # Write output image (object detection output)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image_obscure.jpg')
    cv2.imwrite(output_image_path, image)
    return(output_image_path)


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/image_upload')
def imageHome():
    return render_template('index_upload_and_display_image.html')

@app.route('/video_upload')
def videoHome():
    return render_template('index_upload_and_display_video.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST': #This now supports upload of both image and video file
        uploaded_file = request.files['uploaded-file']
        file_filename = secure_filename(uploaded_file.filename)
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_filename))
 
        session['uploaded_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], file_filename)
 
        if "image" in request.form: #In case of image file, go to image page
            return render_template('index_upload_and_display_image_page2.html')
        elif "video" in request.form: #In case of video file, go to video page
            return render_template('index_upload_and_display_video_page2.html')
        else: #Go home if the form is broken - This shouldn't call but should be replaced by an actual error message
            return render_template('home.html') 
 
@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_file_path', None)
    output_image_path = detect_object(img_file_path)
    obscure_object_path = obscure_object(img_file_path)
    return render_template('show_image.html', user_image = img_file_path, detect_image = output_image_path, obscure_image = obscure_object_path)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/show_live_detect')
def showLiveDetect():
    return render_template('show_live.html')

@app.route('/live_feed_detect')
def live_feed_detect():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/show_live_obscure')
def showLiveObscure():
    return render_template('show_live_obscure.html')

@app.route('/live_feed_obscure')
def live_feed_obscure():
    return Response(gen(VideoCamera(version=1)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/show_video_detect')
def showVideoDetect():
    return render_template('show_video.html')

@app.route('/video_feed_detect')
def video_feed_detect():
    return Response(gen(VideoCamera(session.get('uploaded_file_path', None))),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/show_video_obscure')
def showVideoObscure():
    return render_template('show_video_obscure.html')

@app.route('/video_feed_obscure')
def video_feed_obscure():
    return Response(gen(VideoCamera(session.get('uploaded_file_path', None), version=1)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
 
# flask clear browser cache (disable cache)
# Solve flask cache images issue
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
 
if __name__=='__main__':
    app.run(debug = True)

# Press any key to close the image
#cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
