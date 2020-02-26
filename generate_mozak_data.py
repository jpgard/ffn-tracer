"""
Generate a dataset of neuron images from Mozak datasets.

Creates a set of directories in out_dir for coords and tfrecords representing the data.

usage:

[2d data]

python generate_mozak_data.py \
    --dataset_ids 319215569 327671477 476912429 495358721 507727402 508821490 515548817 \
    515843906 517797791 520260582 521693148 521702225 528890211 529751320 538835830 \
    548268538 550168314 565040416 565298596 565636436 565724110 567856722 568079643 \
    570369389 574690722 583986008 594645586 604897314 646048002 652405209 668585598 \
    671663066 675132228 681189535 712206572 829777525 846611020 933545403 939682871 \
    954441830 957076962 \
    --gs_dir data/gold_standard \
    --img_dir data/img \
    --out_dir ./data/clean-02-2020 \
    --train_data_sampling proportional_by_dataset

[3d data; load from DAP server when img_dir is not provided]
python generate_mozak_data.py \
    --data_dim 3 \
    --dataset_ids 476667707 \
    --gs_dir data/gold_standard \
    --out_dir ./data \
    --train_data_sampling proportional_by_dataset

"""

import argparse
from fftracer.datasets.mozak import MozakDataset2d, MozakDataset3d
from fftracer.datasets import offset_dict_to_csv
import os.path as osp


def main(dataset_ids, gs_dir, out_dir, num_training_coords, coord_margin_xy,
         train_data_sampling, coord_sampling_prob, data_dim, img_dir=None):
    neuron_datasets = dict()
    neuron_offsets = dict()
    for dataset_id in dataset_ids:
        if data_dim == 2:
            dset = MozakDataset2d(dataset_id)
        elif data_dim == 3:
            dset = MozakDataset3d(dataset_id)
        dset.load_data(gs_dir, img_dir)
        neuron_datasets[dataset_id] = dset
        # write data to a tfrecord file
        dset.write_tfrecord(out_dir)
        # write training coordinates (this does work of ffn's build_coordinates.py)
        dset.generate_training_coordinates(out_dir, num_training_coords, coord_margin_xy,
                                           method=train_data_sampling,
                                           coord_sampling_prob=coord_sampling_prob)
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
    parser.add_argument("--num_training_coords",
                        help="number of training coordinates to generate",
                        default=5000,
                        required=False,
                        type=int)
    parser.add_argument("--coord_sampling_prob",
                        type=float,
                        default=0.25,
                        help="probability that each ground-truth pixel is sampled under"
                             " proportional_by_dataset sampling strategy")
    parser.add_argument("--coord_margin_xy",
                        type=int,
                        default=179,
                        help="sampled coordinates must be at least this far from image "
                             "boundaries; use (fov_size_xy - 1 // 2) + delta_xy * "
                             "fov_moves. Note that generally data are not close to "
                             "image boundaries in xy dimension, so this does not "
                             "affect 2D tracing data much and setting to a value "
                             "matching the largest candidate FOV size will result in "
                             "only minimal data loss, if any.")
    parser.add_argument("--train_data_sampling",
                        help="method to use for sampling training data; currently "
                             "'uniform_by_dataset', 'proportional_by_dataset' and "
                             "'balanced_fa' are supported.",
                        required=True)
    parser.add_argument("--data_dim", help="whether to use 2d or 3d data",
                        default=2, type=int, choices=[2, 3])
    args = parser.parse_args()
    main(**vars(args))
