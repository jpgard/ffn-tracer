"""
Train a flood-filling tracer model.

usage:
python train.py --tfrecord_dir ./data/tfrecords --out_dir . --debug
"""

import tensorflow as tf
import os
from fftracer.utils import FEATURE_SCHEMA
import argparse
import skimage


def main(tfrecord_dir, out_dir, debug):
    tfrecord_files = [os.path.join(tfrecord_dir, x)
                      for x in os.listdir(tfrecord_dir)
                      if x.endswith("tfrecord")]
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_files)

    def _parse_image_function(example_proto):
      # Parse the input tf.Example proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, FEATURE_SCHEMA)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    if debug:
        print("[DEBUG] parsed image dataset: {}".format(parsed_image_dataset))
    for image_features in parsed_image_dataset:
        shape = (image_features['shape_x'].numpy()[0],
                 image_features['shape_y'].numpy()[0])
        image_raw = tf.sparse.to_dense(
            image_features['image_raw']).numpy().reshape(shape)
        image_label = tf.sparse.to_dense(
            image_features['image_label']).numpy().reshape(shape)
        if debug:
            print("[DEBUG] saving debug images of raw image and trace")
            skimage.io.imsave("tfrecord_image_raw.jpg", image_raw)
            skimage.io.imsave("tfrecord_image_label.jpg", image_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_dir", help="directory containng tfrecord files",
                        required=True)
    parser.add_argument("--out_dir", help="directory to save to", required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
