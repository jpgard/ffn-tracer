"""
Train a flood-filling tracer model.

usage:
python train.py --tfrecord_dir ./data/tfrecords --out_dir . --debug
"""

import os
from fftracer.utils import FEATURE_SCHEMA
from fftracer.training.input import load_tfrecord_dataset, make_batch_from_dataset, \
    parse_example
import argparse


def main(tfrecord_dir, out_dir, debug):
    tfrecord_files = [os.path.join(tfrecord_dir, x)
                      for x in os.listdir(tfrecord_dir)
                      if x.endswith("tfrecord")]
    parsed_image_dataset = load_tfrecord_dataset(tfrecord_files, FEATURE_SCHEMA)

    for training_batch in make_batch_from_dataset(parsed_image_dataset,
                                                  parse_example_fn=parse_example):
        #TODO(jpgard): make batches and train the model
        pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_dir", help="directory containng tfrecord files",
                        required=True)
    parser.add_argument("--out_dir", help="directory to save to", required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
