import tensorflow as tf
import pickle


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_observation(obs):
    data = {
        'input': _float_feature(obs['input'].numpy().flatten()),
        'output': _int64_feature(obs['output'])
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=data))
    return example_proto.SerializeToString()


def write_obs_tfrecord(dataset, path, seq_len=4, sent_len=200):
    writer = tf.data.experimental.TFRecordWriter(path)
    dataset = dataset.map(tf.io.serialize_tensor)
    writer.write(dataset)
    with open(path + '.pkl', 'wb') as f:
        pickle.dump({'seq_len': seq_len, "sent_len": sent_len}, f)


def read_obs_tfrecord():
    pass
