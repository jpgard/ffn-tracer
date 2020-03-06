"""
Generate a synthetic dataset for testing.

usage:
python generate_synthetic_data.py --out_dir ./synthetic-data

"""
import argparse
from fftracer.datasets.synthetic import SyntheticDataset2D


def main(out_dir, num_training_coords):
    dset = SyntheticDataset2D()
    dset.initialize_synthetic_data_patch(dataset_shape=(1000, 1000), patch_size=(49, 49))
    # write the synthetic data
    dset.write_tfrecord(out_dir)
    # write some synthetic coordinates
    dset.generate_and_write_training_coordinates(out_dir, num_training_coords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", help="directory to save to", required=True)
    parser.add_argument("--num_training_coords", help="number of training coordinates "
                                                      "to generate", type=int,
                        default=100)
    args = parser.parse_args()
    main(**vars(args))
