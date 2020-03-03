"""
A script to copy the corresponding gold standard and image files matching the paired-xy
images in a directory.

This saves from having to manually copy the image and gold standard corresponding to
each dataset, once a set of xy images have been curated.

The dirnames below should be upgraded to command-line arguments in the future!
"""

import os
import re
import shutil

xy_pairs_dirname = "/Users/jpgard/Documents/github/ffn-tracer/data/clean-02-2020/xy-pairs"
gs_source_dirname = "/Users/jpgard/Documents/github/ffn-tracer/data/gold_standard"
gs_dest_dirname = "/Users/jpgard/Documents/github/ffn-tracer/data/clean-02-2020" \
                  "/gold_standard"
img_source_dirname = "/Users/jpgard/Documents/github/ffn-tracer/data/img"
img_dest_dirname = "/Users/jpgard/Documents/github/ffn-tracer/data/clean-02-2020/img"

files = [f for f in os.listdir(xy_pairs_dirname)
         if os.path.isfile(os.path.join(xy_pairs_dirname, f))]
for f in os.listdir(xy_pairs_dirname):
    print(f)
    re_match = re.match("(\d+)-xy.png", f)
    if re_match:
        dataset_id = re_match.group(1)
        img_filename = ".".join([dataset_id, "png"])
        shutil.copy(os.path.join(img_source_dirname, img_filename),
                    img_dest_dirname)
        gs_filename = ".".join([dataset_id, "swc"])
        shutil.copy(os.path.join(gs_source_dirname, gs_filename),
                    gs_dest_dirname)
