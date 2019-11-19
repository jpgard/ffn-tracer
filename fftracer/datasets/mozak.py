import skimage

import numpy as np
import os.path as osp

from fftracer.datasets import PairedDataset2d
from mozak.utils.connectors import ImageAPIConnector
from mozak.datasets.gold_standard import MozakGoldStandardTrace
from mozak.datasets.trace import nodes_and_edges_to_trace
from fftracer.utils import VALID_IMAGE_EXTENSIONS
import glob
import tensorflow as tf
from fftracer.utils.features import _int64_feature, _bytes_feature


class MozakDataset2d(PairedDataset2d):

    def load_x_from_dap(self):
        imshape = (76 * 10 ** 2, 76 * 10 ** 2)
        center = list(map(lambda x: x // 2, imshape))
        # get the image data for x
        img_api = ImageAPIConnector()
        img_bytes = img_api.request_image(self.dataset_id, center=center, size=imshape)
        self.x = np.asarray(img_bytes.convert("L"))

    def load_x_from_file(self, img_dir):
        x_file = [f for f in glob.glob(osp.join(img_dir, self.dataset_id + ".*"))
        if f.endswith(VALID_IMAGE_EXTENSIONS)][0]
        print("[INFO] loading image from {}".format(x_file))
        self.x = skimage.io.imread(x_file, as_gray=True)

    def load_data(self, gs_dir, data_dir=None):
        """
        Load the image data and the gold standard data.
        :param gs_dir: directory containing gold-standard trace.
        :return: None.
        """
        if not data_dir:
            self.load_x_from_dap()
        else:
            self.load_x_from_file(data_dir)

            # get the mask data for y
        gs = MozakGoldStandardTrace(self.dataset_id, gs_dir)
        gs.fetch_trace()
        # create "soft labels" map
        self.y = nodes_and_edges_to_trace(gs.nodes, gs.edges, imshape=self.x.shape,
                                          trace_value=0.95, pad_value=0.05)
        self.check_xy_shapes_match()
        return

    def serialize_example(self):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'shape_x': _int64_feature(self.shape[0]),
            'shape_y': _int64_feature(self.shape[1]),
            'seed_x': _int64_feature(self.seed.x),
            'seed_y': _int64_feature(self.seed.y),
            'seed_z': _int64_feature(self.seed.z),
            'image_raw': tf.train.Feature(
                int64_list=tf.train.Int64List(value=self.x.flatten().tolist())
            ),
            'image_label': tf.train.Feature(
                float_list=tf.train.FloatList(value=self.y.flatten().tolist())
            ),
        }
        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
