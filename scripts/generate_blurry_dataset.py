"""
Generate a dataset where the 'images' consist of darker, blurrier versions of the
ground-truth images.

Usage:


python scripts/generate_blurry_dataset.py \
    --dataset_ids 319215569 327671477 476912429 \
    --gs_dir data/gold_standard \
    --out_dir ./data/blurry \
    --train_data_sampling proportional_by_dataset
"""

import argparse
from fftracer.datasets.blurry import BlurryDataset2D
# from fftracer.datasets import offset_dict_to_csv
import os.path as osp

from PIL import Image

# Disable PIL DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None


def main(dataset_ids, out_dir, gs_dir, train_data_sampling):
    for dataset_id in dataset_ids:
        dset = BlurryDataset2D(dataset_id)
        dset.load_and_generate_data(gs_dir)
        # write the synthetic data
        dset.write_tfrecord(out_dir)
        # write some synthetic coordinates
        dset.generate_and_write_training_coordinates(
            out_dir,
            # num_training_coords # TODO(jpgard): implement
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_ids", help="dataset id", nargs="+", required=True)
    parser.add_argument("--out_dir", help="directory to save to", required=True)
    parser.add_argument("--gs_dir", help="directory containing gold-standard traces",
                        required=True)
    parser.add_argument("--train_data_sampling",
                        help="method to use for sampling training data; currently "
                             "'uniform_by_dataset', 'proportional_by_dataset' and "
                             "'balanced_fa' are supported.",
                        required=True)
    args = parser.parse_args()
    main(**vars(args))
