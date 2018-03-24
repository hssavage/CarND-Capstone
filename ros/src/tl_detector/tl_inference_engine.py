'''
###############################################################################
# tl_inference_engine.py                                                      #
# -----------                                                                 #
# This module contains the source for the running the tensorflow model in an  #
# inference engeine.  Not that the model needs to be generated using the same #
# tensorflow version as running on carla ie 1.3.                               #
#                                                                             #
# Change Log:                                                                 #
# -----------                                                                 #
# +--------------------+---------------+------------------------------------+ #
# | Date               | Author        | Description                        | #
# +--------------------+---------------+------------------------------------+ #
# | 3/18/2018          | Jason Clemons | Initial pass on the code,          | #
# |                    |               | Needs a lot of clean up but a      | #
# |                    |               | good starting poc and example      | #
# +--------------------+---------------+------------------------------------+ #
# | 3/18/2018          | Jason Clemons | Added filter based on area and     | #
# |                    |               | Added filter based on histogram    | #
# |                    |               | Also translating to ros class ids  | #
# +--------------------+---------------+------------------------------------+ #
###############################################################################
'''



import numpy as np
import os
import sys
import tensorflow as tf
import cv2
from collections import defaultdict
from io import StringIO
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
from styx_msgs.msg import TrafficLightArray, TrafficLight



def draw_bounding_boxes(image, boxes, classes,color_list ,thickness=4):
'''
Draw bounding boxes with the passed in the color list based on the classes
'''


    #THis function was written assuming PIL but we are uisng opencv so
    # convert the image to a PIL image
    im_pil = Image.fromarray(image)
    
    """Draw bounding boxes on the image"""

    #Taken from the class example of drawing bounding boxes
    draw = ImageDraw.Draw(im_pil)

    for i in range(len(boxes)):
#        print('Drawing ', boxes [i,...])
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
 #       print("Classes: ", class_id)
        color = color_list[class_id]
        #COLOR_LIST[class_id-i]
#        print('color: ', type(color))
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

    #COnvert back to numpy array/cv format  
    out_image = np.array(im_pil)

    return out_image


#This class holds the tf session and graph to do inference as needed
class DetectionInferenceEngine(object):
'''
This class holds an inference engine that holds the tensorflow objects for doing inference
'''


    def __init__(self,model_path='models/mobilenet_ssd/frozen_inference_graph.pb', confidence_thr= 0.02, min_area = 1000, max_area = 25200,class_dict= None, color_list=None):


        #Store the graph object we will load the model into
        self.detection_graph = tf.Graph()

        #load the model
        self.load_graph(model_path)

        #Setup a threshold for accepting bounding box
        self.confidence_thr = confidence_thr

        self.min_area = min_area
        self.max_area = max_area

        #A list of colors to draw with based on classification
        if(color_list == None):
            self.color_list = ['red','yellow','green','gray','white']#['white','green', 'yellow', 'red','gray']
        else:
            self.color_list = color_list

        #CLass Names
        if class_dict == None:
            self.class_dict = {0: 'Unknown', 1: 'Green', 2:'Yellow', 3:'Red', 4:'Off' }
        else:
            self.class_dict = class_dict


    #Use tensorflow to load the frozen model
    def load_graph(self, model_path):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session()

    # A utility function to get numpy array from PIl image
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


    #Run through the inference
    def run_inference(self, image):

        #Use our graph 
        with self.detection_graph.as_default():
            tensor_dict = {}

            #Get the tf nodes so we can get the nodes we need based on name
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)       

            #grab the input
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = self.sess.run(tensor_dict, feed_dict={image_tensor: image})       


            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]


            #We need to filter these results based on confidence
            output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes'] = self.filter_boxes(self.confidence_thr, output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes'])
 
#            print('shape: ',image.shape)

            #Convert the bounding box results which are normalized coordinates [0 to 1) to actual image coordinates 
            height, width = image.shape[1:3]
            output_dict['detection_boxes'] = self.to_image_coords(output_dict['detection_boxes'], height, width)
            output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes'] = self.filter_boxes_by_area(self.min_area,  self.max_area, output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes'])

            output_dict['detection_classes'] = self.translate_classes(output_dict['detection_classes'])

            predicted_light_state = self.predict_current_light_state(output_dict['detection_classes'])

        #Return the dictionary      
        return output_dict, predicted_light_state

    #Translating 
    def translate_classes(self, classes):
    '''
    Translate the ids.  For training do not use a 0 index so we have to translate for talking to ros
    '''

        new_classes = []

        for current_class in classes:
            if current_class == 1:
                new_classes.append(TrafficLight.GREEN)
            if current_class == 2:
                new_classes.append(TrafficLight.YELLOW)
            if current_class == 3:
                new_classes.append(TrafficLight.RED)
            if current_class == 4:
                new_classes.append(TrafficLight.UNKNOWN)

        return new_classes

    def predict_current_light_state(self, classes):
    '''
    Use a histogram to predict the light state based number of each class in the image
    '''

        predicted_state = TrafficLight.UNKNOWN
        if len(classes) > 0:
            histogram = [0,0,0,0,0]
            for light in classes:
                histogram[light] +=1
            max_value = max(histogram)
            max_index = histogram.index(max_value)
            predicted_state = max_index


        return predicted_state


    def filter_boxes(self,min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]

        return filtered_boxes, filtered_scores, filtered_classes


    def filter_boxes_by_area(self,min_area,max_area, boxes, scores, classes):
        """Return boxes within area limits`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            bot, left, top, right = boxes[i, ...]
            area = abs((bot-top) * (right - left))
            print('Area: ', area)
            if area >= min_area and area <= max_area:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]

        return filtered_boxes, filtered_scores, filtered_classes



    def to_image_coords(self,boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

