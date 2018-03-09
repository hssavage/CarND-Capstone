import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import ops as utils_ops



flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_yaml_path', '', 'Path to input yaml file')
flags.DEFINE_integer('max_num_examples', 6000, 'Maxt number of images to pull')
flags.DEFINE_string('image_base_path', '', 'base path to images')

FLAGS = flags.FLAGS
  
MODEL_NAME = 'models/mobilenet_ssd'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'bosch_traffic_light.pbtxt')

NUM_CLASSES = 3

def filter_boxes(min_score, boxes, scores, classes):
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

def to_image_coords(boxes, height, width):
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


class DetectionInferenceEngine():
	def __init__(self):
		self.detection_graph = tf.Graph()
		self.load_graph()
	def load_graph(self):
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
			self.sess = tf.Session()

	def load_image_into_numpy_array(self, image):
		(im_width, im_height) = image.size
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
	def run_inference(self, image):

		with self.detection_graph.as_default():
			tensor_dict = {}

			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)		
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			# Run inference
			output_dict = self.sess.run(tensor_dict, feed_dict={image_tensor: image})		
			# all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
      
		return output_dict

  
  
def main(_):


	inference_engine = DetectionInferenceEngine()
	image_path = './432522.png'
	image = Image.open(image_path)
	# the array based representation of the image will be used later in order to prepare the
	# result image with boxes and labels on it.
	image_np = inference_engine.load_image_into_numpy_array(image)
	print('image_np shape ', image_np.shape)
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image_np, axis=0)
	# Actual detection.
	print('image_np_expanded shape ', image_np_expanded.shape)

	output_dict = inference_engine.run_inference(image_np_expanded)




	print("Dict Result: ", output_dict)
	confidence_cutoff = 0.05
    # Filter boxes with a confidence score less than `confidence_cutoff`
	boxes, scores, classes = filter_boxes(confidence_cutoff, output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes'])

    # The current box coordinates are normalized to a range between 0 and 1.
    # This converts the coordinates actual location on the image.
	width, height = image.size
	box_coords = to_image_coords(boxes, height, width)

	print('box coordinates: ', box_coords)


	return
if __name__ == '__main__':
    tf.app.run()


#python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /workspace/models/mobilenet_ssd/bosch_traffic_dataset.config --trained_checkpoint_prefix /workspace/models/mobilenet_ssd/train/  --output_directory /workspace/models/mobilenet_ssd/output_inference_graph.pb
