import os
from ruamel.yaml import YAML
import tensorflow as tf
from object_detection.utils import dataset_util
import io
import numpy as np
import PIL.Image
import json


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_file_path', '', 'Path to input file')
flags.DEFINE_integer('max_num_examples', 6000, 'Maxt number of images to pull')
flags.DEFINE_string('image_base_path', '', 'base path to images')

FLAGS = flags.FLAGS


LABEL_DICT =  {
    "Green" : 1,
    "Red" : 3,
    "GreenLeft" : 1,
    "GreenRight" : 1,
    "RedLeft" : 3,
    "RedRight" : 3,
    "Yellow" : 2,
    "Off" : 4,
    "RedStraight" : 3,
    "GreenStraight" : 1,
    "GreenStraightLeft" : 1,
    "GreenStraightRight" : 1,
    "RedStraightLeft" : 3,
    "RedStraightRight" : 3
    }




def get_dict(filename):
    

    with open(filename, 'r', encoding = "utf-8") as yaml_file:

        yaml=YAML(typ='safe')
        data_set_dict = yaml.load(yaml_file)
    print(len(data_set_dict))

    return data_set_dict


def load_sloth_json_file(filename):
    json_file = open(filename)
    json_str = json_file.read()
    sloth_dict = json.loads(json_str)

    return sloth_dict


def get_tf_example(example_data, image_base_path, verbose =False):
#    height = 720 # Image height
#    width = 1280 # Image width

#    _, file_extension = os.path.splitext(example_data['path'])
    #print('filename_base', filename_base, ' ext: ', file_extension)
    filename = example_data['filename'] # Filename of the image. Empty if image is not from file
	
    if not image_base_path =='':
        base_image_name = os.path.basename(filename)
        filename = image_base_path + '/'+ base_image_name

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image_data = fid.read()
    filename=filename.encode()

    encoded_image_data_io = io.BytesIO(encoded_image_data)

    image = PIL.Image.open(encoded_image_data_io)
    width, height = image.size
    if verbose:

        print( width, 'x', height)
    #encoded_image_data = None # Encoded image bytes

    _, file_extension = os.path.splitext(filename)
    file_extension = file_extension[1:]
    if verbose:
        print('file_extension: ', file_extension)

    image_format = file_extension
    if(image_format == b'jpg'):
        image_format = b'jpeg'

    #'png'.encode()
    #file_extension.encode() # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
            # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
            # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)


    for annotation in example_data['annotations']:
        xmins.append(annotation['x']/width)        
        xmaxs.append((annotation['x']+annotation['width'])/width)        
        ymins.append(annotation['y']/height)        
        ymaxs.append((annotation['y']+annotation['height'])/height)        
        classes_text.append(annotation['id'].encode())
        classes.append(int(LABEL_DICT[annotation['id']]))


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):

    sloth_dict = load_sloth_json_file(FLAGS.input_file_path)

    print('input_path: ',FLAGS.input_file_path )

#    print(get_tf_example(data_set_dict[0]))
    print('output_path: ',FLAGS.output_path )
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)



    # TODO: Write code to read in your dataset to examples variable
    loop_count = 0

    for example in sloth_dict:
        #print("Example: ", example)
        tf_example = get_tf_example(example,FLAGS.image_base_path, verbose=True)
#        print(tf_example)
        writer.write(tf_example.SerializeToString())
        loop_count = loop_count +1
        if(loop_count >=FLAGS.max_num_examples):
            break
    writer.close()
    print ("Converted: ", loop_count, " examples")
    return





if __name__ == '__main__':
    tf.app.run()
