"""
Script to read SWC files and images, and produce side-by-side plots of image and ground
truth arrays for inspection.
"""

import os
import re
import glob
import numpy as np

from absl import flags
from absl import app
from skimage.io import imread, imsave
from PIL import Image


from mozak.datasets.gold_standard import MozakGoldStandardTrace
from mozak.datasets.trace import nodes_and_edges_to_trace

FLAGS = flags.FLAGS

flags.DEFINE_string("swc_dir", None, "directory to seach for SWC files.")
flags.DEFINE_string("img_dir", None, "directory to search for images.")
flags.DEFINE_string("img_ext", "png", "file extension of the input/output images.")
flags.DEFINE_string("out_dir", None, "directory to save results to.")

# Disable PIL DecompressionBomError for large images
Image.MAX_IMAGE_PIXELS = None


def main(argv):
    swc_files = [f for f in os.listdir(FLAGS.swc_dir) if f.endswith(".swc")]
    for swc in swc_files:
        dataset_id = re.match("(\d+)\.swc", swc).group(1)
        # Read the input image
        try:
            x_file = [f for f in glob.glob(os.path.join(FLAGS.img_dir, dataset_id + ".*"))
                      if f.endswith(FLAGS.img_ext)][0]
        except:  # No input image available for this trace; move to next dataset.
            print("[INFO] no input image in {} for dataset_id {}; skipping".format(
                FLAGS.img_dir, dataset_id))
            continue
        print("[INFO] loading image from {}".format(x_file))
        x = imread(x_file, as_gray=True)
        x = np.array(x)
        # Read the gold standard trace
        gs = MozakGoldStandardTrace(dataset_id, FLAGS.swc_dir)
        gs.fetch_trace()
        y = nodes_and_edges_to_trace(gs.nodes, gs.edges, imshape=x.shape,
                                     trace_value=255,
                                     pad_value=0)
        assert y.shape == x.shape, "shape mismatch between x and y arrays"
        xy_paired_array = np.concatenate([x,y], axis=1)
        out_file = "{dataset_id}-xy.{ext}".format(dataset_id=dataset_id, ext=FLAGS.img_ext)
        out_fp = os.path.join(FLAGS.out_dir, out_file)
        print("[INFO] writing file to {}".format(out_fp))
        im = Image.fromarray(xy_paired_array)
        # im.save(out_fp)
        imsave(out_fp, xy_paired_array)
        # TODO: copy image, gs_file, and output image into some destination directory


if __name__ == "__main__":
    app.run(main)
