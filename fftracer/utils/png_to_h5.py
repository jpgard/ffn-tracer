"""

Forked from ffn/utils/png_to_h5.py.

Converts PNG files from the working directory into a
HDF5 volume.

Usage:
  ./png_to_h5.py output_filename.h5
"""

import glob
import sys

import h5py
import numpy as np
import imageio

assert len(sys.argv) >= 2

png_files = glob.glob('*.png')
png_files.sort()
images = [imageio.imread(i) for i in png_files]
images = np.array(images)
with h5py.File(sys.argv[1], 'w') as f:
    f.create_dataset('raw', data=images, compression='gzip')
