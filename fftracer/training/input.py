"""
Functions for processing model input.
"""

import tensorflow as tf

def load_tfrecord_dataset(tfrecord_files, feature_schema):
    """
    Load and parse the dataset in tfrecord files.
    :param tfrecord_files: list of files containg tfrecords
    :param feature_schema: dict of {feature_name, tf.io.<feature_type>}
    :return: the dataset parsed according to the specified feature schema.
    """
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_files)

    def _parse_image_function(example_proto):
      # Parse the input tf.Example proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, feature_schema)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    return parsed_image_dataset


def make_batch_from_dataset(dataset, parse_example_fn):
    """
    Create a batch of training instances from a tf.Example of a complete neuron.
    :param dataset:
    :param parse_example_fn: a function which returns a dict with keys "x", "y", "seed".
    :return:
    """
    for element in dataset:
        parsed_example = parse_example_fn(element)
    return


def parse_example(example: dict):
    """
    Parse an example in the default format.
    :param example: an element of a tensorflow.python.data.ops.dataset_ops.MapDataset
    :return: dict with keys "x", "y", "seed" with corresponding Tensors.
    """
    shape = tf.concat([example['shape_x'], example['shape_y']], axis=-1)
    image_raw = tf.reshape(tf.sparse.to_dense(example['image_raw']), shape)
    image_label = tf.reshape(tf.sparse.to_dense(example['image_label']), shape)
    seed = tf.concat([example['seed_x'], example['seed_y'], example['seed_z']], axis=-1)
    #     to see images:
    #     print("[DEBUG] saving debug images of raw image and trace")
    #     skimage.io.imsave("tfrecord_image_raw.jpg", image_raw.numpy())
    #     skimage.io.imsave("tfrecord_image_label.jpg", image_label.numpy())
    return {"x": image_raw, "y": image_label, "seed": seed}
