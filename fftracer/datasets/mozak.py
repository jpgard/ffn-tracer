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
                                          trace_value=1 - self.pom_pad,
                                          pad_value=self.pom_pad)
        self.check_xy_shapes_match()
        return

    def serialize_example(self):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'shape_x': _int64_feature([self.shape[0]]),
            'shape_y': _int64_feature([self.shape[1]]),
            'seed_x': _int64_feature([self.seed.x]),
            'seed_y': _int64_feature([self.seed.y]),
            'seed_z': _int64_feature([self.seed.z]),
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

    def write_tfrecord(self, out_dir):
        tfrecord_filepath = osp.join(out_dir, self.dataset_id + ".tfrecord")
        with tf.io.TFRecordWriter(tfrecord_filepath) as writer:
            example = self.serialize_example()
            writer.write(example)

    def sample_training_coordinates(self, n: int):
        assert not np.all(self.y == self.pom_pad), \
            "cannot sample coordinates from empty map"
        # take a random sample of positive labels
        candidate_indices = np.argwhere(self.y == 1 - self.pom_pad)
        sample_ix = np.random.choice(candidate_indices.shape[0], size=n, replace=False)
        sample = candidate_indices[sample_ix, :]
        return sample

    def generate_training_coordinates(self, out_dir, n):
        coords = self.sample_training_coordinates(n)
        tfrecord_filepath = osp.join(out_dir, self.dataset_id + "_coords.tfrecord")
        record_options = tf.io.TFRecordOptions(
            tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
        with tf.io.TFRecordWriter(tfrecord_filepath,
                                         options=record_options) as writer:
            for coord in coords:
                x,y = coord.tolist()
                coord_zyx = [0, y, x] # store in reverse to match ffn formatting
                coord = tf.train.Example(features=tf.train.Features(feature=dict(
                    center=_int64_feature(coord_zyx),
                    label_volume_name=_bytes_feature([self.dataset_id.encode('utf-8')])
                )))
                writer.write(coord.SerializeToString())
