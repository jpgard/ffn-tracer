"""
Train a flood-filling tracer model.

usage:
python train.py --tfrecord_dir ./data/tfrecords \
    --out_dir . --coordinate_dir ./data/coords
"""

import os

from fftracer.utils import features
from fftracer import utils
from fftracer.training import input
from fftracer.training.models.model import FFNTracerModel
import argparse
import tensorflow as tf


def main(tfrecord_dir, out_dir, debug, coordinate_dir, batch_size, epochs):
    volume_map = input.load_tfrecord_dataset(tfrecord_dir,
                                             utils.features.FEATURE_SCHEMA)
    model = FFNTracerModel(deltas=[8, 8, 0], batch_size=batch_size)
    volume_name = "507727402"

    # Fetch sizes of images and labels

    label_size = features.get_image_shape(volume_map, volume_name).numpy()
    image_size = features.get_image_shape(volume_map, volume_name).numpy()

    # TODO(jpgard): verify that this is the size of the input images, not of the
    #  training frames
    label_radii = (label_size // 2).tolist()
    label_size = label_size.tolist()
    image_radii = (image_size // 2).tolist()
    image_size = image_size.tolist()

    # Fetch a single coordinate and volume name from a queue reading the
    # coordinate files or from saved hard/important examples
    coord, volname = input.load_patch_coordinates(coordinate_dir)
    labels = input.load_from_numpylike_2d(coord, volname, shape=[1, 49, 49],
                                          volume_map=volume_map,
                                          feature_name='image_label')
    patch = input.load_from_numpylike_2d(coord, volname, shape=[1, 49, 49],
                                         volume_map=volume_map, feature_name='image_raw')
    import ipdb;ipdb.set_trace()
    # TODO(jpgard): continue following logic of ffn training here; see ffn train.py L#381
    for epoch in range(epochs):
        batches = input.get_batch(volume_map,
                                  parse_example_fn=input.parse_example)
        # for batch in
        # TODO(jpgard): make batches and train the model
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_dir", help="directory containng tfrecord files of "
                                               "labeled input data volumes",
                        required=True)
    parser.add_argument("--coordinate_dir", help="directory containng tfrecord files of "
                                                 "patch coodinates",
                        required=True)
    parser.add_argument("--out_dir", help="directory to save to", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
