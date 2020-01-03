"""
Generate a dataset of neuron images from Mozak datasets.

usage:
python generate_mozak_data.py \
    --dataset_ids 507727402 521693148 522442346 529751320 565040416 565298596 565636436 \
        565724110 570369389 319215569 397462955 476667707 476912429 495358721 508767079 \
        508821490 515548817 515843906 518298467 518358134 518784828 520260582 521693148 \
        521702225 522442346 541830986 548268538 550168314 \
    --gs_dir data/gold_standard \
    --img_dir data/img \
    --seed_csv data/seed_locations/seed_locations.csv \
    --out_dir ./data \
    --num_training_coords 2500 \
    --coord_margin 171
"""

import argparse
from fftracer.datasets.mozak import MozakDataset2d
from fftracer.datasets import SeedDataset, offset_dict_to_csv
import os.path as osp


def main(dataset_ids, gs_dir, seed_csv, out_dir, num_training_coords, coord_margin,
    img_dir=None):
    seeds = SeedDataset(seed_csv)
    neuron_datasets = dict()
    neuron_offsets = dict()
    for dataset_id in dataset_ids:
        seed = seeds.get_seed_loc(dataset_id)
        dset = MozakDataset2d(dataset_id, seed)
        dset.load_data(gs_dir, img_dir)
        neuron_datasets[dataset_id] = dset
        # write data to a tfrecord file
        dset.write_tfrecord(out_dir)
        # write training coordinates (this does work of ffn's build_coordinates.py)
        dset.generate_training_coordinates(out_dir, num_training_coords, coord_margin)
        # save the offets
        neuron_offsets[dataset_id] = dset.fetch_mean_and_std()
        del dset
    # write offsets to csv
    offset_dict_to_csv(neuron_offsets, fp=osp.join(out_dir, "offsets", "offsets.csv"))
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
    parser.add_argument("--num_training_coords", help="number of training coordinates "
                                                      "to generate",
                        required=True, type=int)
    parser.add_argument("--coord_margin", type=int,
                        help="sampled coordinates must be at least this far from image "
                             "boundaries (set to max fov_size // 2")
    args = parser.parse_args()
    main(**vars(args))
