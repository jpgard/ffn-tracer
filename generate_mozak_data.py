"""
Generate a dataset of neuron images from Mozak datasets.

usage:
python generate_mozak_data.py \
    --dataset_ids 507727402 \
    --gs_dir data/gold_standard \
    --img_dir data/img \
    --seed_csv data/seed_locations/seed_locations.csv \
    --out_dir data/tfrecords
"""

import argparse
from fftracer.datasets.mozak import MozakDataset2d
from fftracer.datasets import SeedDataset
import tensorflow as tf
import os.path as osp


def main(dataset_ids, gs_dir, seed_csv, out_dir, img_dir=None):
    seeds = SeedDataset(seed_csv)
    neuron_datasets = dict()
    for dataset_id in dataset_ids:
        seed = seeds.get_seed_loc(dataset_id)
        dset = MozakDataset2d(dataset_id, seed)
        dset.load_data(gs_dir, img_dir)
        neuron_datasets[dataset_id] = dset
        # write to a tfrecord file
        tfrecord_filepath = osp.join(out_dir, dataset_id + ".tfrecord")
        with tf.io.TFRecordWriter(tfrecord_filepath) as writer:
            example = dset.serialize_example()
            writer.write(example)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_ids", help="dataset id", nargs="+", required=True)
    parser.add_argument("--out_dir", help="directory to save to", required=True)
    parser.add_argument("--gs_dir", help="directory containing gold-standard traces",
                        required=True)
    parser.add_argument("--img_dir", help="directory containing raw input images",
                        required=False)
    parser.add_argument("--seed_csv", help="csv file containing seed locations",
                        required=True)
    args = parser.parse_args()
    main(**vars(args))
