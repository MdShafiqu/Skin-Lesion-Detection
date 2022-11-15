# Import packages
from ctypes.util import find_library
import ctypes as ct
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

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

# Number of classes the object detector can identify
NUM_CLASSES = 11

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
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

#Define EvoIRFrameMetadata structure for additional frame infos
class EvoIRFrameMetadata(ct.Structure):
     _fields_ = [("counter", ct.c_uint),
                 ("counterHW", ct.c_uint),
                 ("timestamp", ct.c_longlong),
                 ("timestampMedia", ct.c_longlong),
                 ("flagState", ct.c_int),
                 ("tempChip", ct.c_float),
                 ("tempFlag", ct.c_float),
                 ("tempBox", ct.c_float),
                 ]

class VideoCamera(object):
    def __init__(self, source=0):
        self.video = cv2.VideoCapture(source)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
         libir = ct.CDLL("c:\\lib\\irDirectSDK\\sdk\\x64\\libirimager.dll")
         #path to config xml file
         pathXml = ct.c_char_p(b'C:/lib/irDirectSDK/generic.xml')
         # init vars
         pathFormat = ct.c_char_p(b'C:/lib/irDirectSDK')
         pathLog = ct.c_char_p(b'logfilename')
         palette_width = ct.c_int()
         palette_height = ct.c_int()

         thermal_width = ct.c_int()
         thermal_height = ct.c_int()

         serial = ct.c_ulong()

         # init EvoIRFrameMetadata structure
         metadata = EvoIRFrameMetadata()

         # init lib
         ret = libir.evo_irimager_usb_init(pathXml, pathFormat, pathLog)
         #if ret != 0:
              #print("error at init")
              #exit(ret)

         # get the serial number
         ret = libir.evo_irimager_get_serial(ct.byref(serial))
         #print('serial: ' + str(serial.value))

         # get thermal image size
         libir.evo_irimager_get_thermal_image_size(ct.byref(thermal_width), ct.byref(thermal_height))
         #print('thermal width: ' + str(thermal_width.value))
         #print('thermal height: ' + str(thermal_height.value))

         # init thermal data container
         np_thermal = np.zeros([thermal_width.value * thermal_height.value], dtype=np.uint16)
         npThermalPointer = np_thermal.ctypes.data_as(ct.POINTER(ct.c_ushort))

         # get palette image size, width is different to thermal image width duo to stride alignment!!!
         libir.evo_irimager_get_palette_image_size(ct.byref(palette_width), ct.byref(palette_height))
         #print('palette width: ' + str(palette_width.value))
         #print('palette height: ' + str(palette_height.value))

         # init image container
         np_img = np.zeros([palette_width.value * palette_height.value * 3], dtype=np.uint8)
         npImagePointer = np_img.ctypes.data_as(ct.POINTER(ct.c_ubyte))
         #get thermal and palette image with metadat
         ret = libir.evo_irimager_get_thermal_palette_image_metadata(thermal_width, thermal_height, npThermalPointer, palette_width, palette_height, npImagePointer, ct.byref(metadata))
         frame=np_img
         frame=frame.reshape(palette_height.value, palette_width.value, 3)
         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         frame_expanded = np.expand_dims(frame_rgb, axis=0)
         # Perform the actual detection by running the model with the image as input
         (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: frame_expanded})
         vis_util.visualize_boxes_and_labels_on_image_array(
              frame,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=2,
              min_score_thresh=0.25,
              skip_labels=True,
              skip_scores=True)
         ret, jpeg = cv2.imencode('.jpg', frame)
         return jpeg.tobytes()
         if ret != 0:
              print('error on evo_irimager_get_thermal_palette_image ' + str(ret))
         #continue
		
     

        # clean shutdown
         #libir.evo_irimager_terminate()


#video.release()
cv2.destroyAllWindows()

    
