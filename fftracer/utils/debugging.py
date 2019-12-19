from PIL import Image
import os
import numpy as np


def write_patch_and_label_to_img(patch, label, unique_id, dirname=None):
    if not dirname:
        dirname = os.getcwd()
    outfile = "patch_and_label_{}.png".format(unique_id)
    patch = patch.astype(np.uint8)
    label = label.astype(np.uint8)
    patch_and_label = np.concatenate([patch, label], axis=1)
    # write these arrays to image files (optionally, concatenate them first).
    img = Image.fromarray(patch_and_label, 'L')
    img.save(os.path.join(dirname, outfile))
    return
