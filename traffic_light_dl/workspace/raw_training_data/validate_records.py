
import tensorflow as tf

def validate_dataset(filenames, reader_opts=None):
    """
    Attempt to iterate over every record in the supplied iterable of TFRecord filenames
    :param filenames: iterable of filenames to read
    :param reader_opts: (optional) tf.python_io.TFRecordOptions to use when constructing the record iterator
    """
    i = 0
    for fname in filenames:
        print('validating ', fname)

        record_iterator = tf.python_io.tf_record_iterator(path=fname, options=reader_opts)
        try:
            for _ in record_iterator:
                i += 1
        except Exception as e:
            print('error in {} at record {}'.format(fname, i))
            print(e)


if __name__ == '__main__':
    filenames = ['/workspace/data/bosch_traffic_light_train.record','/workspace/data/bosch_traffic_light_val.record']
    validate_dataset(filenames)