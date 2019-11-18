"""
Generate a dataset of neuron images from Mozak datasets.

usage:
python scripts/generate_mozak_data.py --dataset_ids 507727402 --gs_dir data/gold_standard
"""

import argparse
from fftracer.datasets.mozak import MozakDataset2d
from fftracer.datasets import SeedDataset

def main(dataset_ids, out_dir, gs_dir, seed_csv):
    seeds = SeedDataset(seed_csv)
    neuron_datasets = dict()
    for dataset_id in dataset_ids:
        seed = seeds.get_seed_loc(dataset_id)
        dset = MozakDataset2d(dataset_id, seed)
        dset.load_data(gs_dir)
        neuron_datasets[dataset_id] = dset
    # TODO(jpgard): generate individual training examples to match training procedure
    #  in original ffn paper and write in a tensorflow format to out_dir.
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_ids", help="dataset id", nargs="+", required=True)
    parser.add_argument("--out_dir", help="directory to save to", required=True)
    parser.add_argument("--gs_dir", help="directory containing gold-standard traces",
                        required=True)
    parser.add_argument("--seed_csv", help="csv file containing seed locations",
                        required=True)
    args = parser.parse_args()
    main(**vars(args))
