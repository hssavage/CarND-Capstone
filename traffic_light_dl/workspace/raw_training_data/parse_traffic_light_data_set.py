import os
from ruamel.yaml import YAML
import tensorflow as tf
from object_detection.utils import dataset_util
import io
import numpy as np
import PIL.Image

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_yaml_path', '', 'Path to input yaml file')
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
    "off" : 3,
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


def get_tf_example(example_data, image_base_path):
    height = 720 # Image height
    width = 1280 # Image width

#    _, file_extension = os.path.splitext(example_data['path'])
    #print('filename_base', filename_base, ' ext: ', file_extension)
    filename = example_data['path'] # Filename of the image. Empty if image is not from file
	
    if not image_base_path =='':
        base_image_name = os.path.basename(filename)
        filename = image_base_path + '/'+ base_image_name

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image_data = fid.read()
    filename=filename.encode()

    #encoded_image_data = io.BytesIO(encoded_png)
    #image = PIL.Image.open(encoded_image_data)

    #encoded_image_data = None # Encoded image bytes

    #_, file_extension = os.path.splitext(filename)
    #file_extension = file_extension[1:]
    #print('file_extension: ', file_extension)
    image_format = 'png'.encode()
    #file_extension.encode() # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
            # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
            # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)



    for box in example_data['boxes']:
        xmins.append(box['x_min']/width)        
        xmaxs.append(box['x_max']/width)        
        ymins.append(box['y_min']/height)        
        ymaxs.append(box['y_max']/height)        
        classes_text.append(box['label'].encode())
        classes.append(int(LABEL_DICT[box['label']]))


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



    #return [xmins, xmaxs, ymins, ymaxs, classes_text, classes]

def main(_):


    data_set_dict= get_dict(FLAGS.input_yaml_path)
    print('yaml_path: ',FLAGS.input_yaml_path )

#    print(get_tf_example(data_set_dict[0]))
    print('output_path: ',FLAGS.output_path )
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)



    # TODO: Write code to read in your dataset to examples variable
    loop_count = 0

    for example in data_set_dict:
        tf_example = get_tf_example(example,FLAGS.image_base_path)
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
