import numpy as np
import os.path as osp
import glob

from skimage.io import imread

from fftracer.datasets import PairedDataset2d
from mozak.utils.connectors import ImageAPIConnector
from mozak.datasets.gold_standard import MozakGoldStandardTrace
from mozak.datasets.trace import nodes_and_edges_to_trace
from fftracer.utils import VALID_IMAGE_EXTENSIONS


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
        x = imread(x_file, as_gray=True)
        self.x = np.array(x)

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

    def sample_training_coordinates(self, n: int, coord_margin: int):
        """Create a list of (x,y) coordinates for training. Only coordinates with
        positive labels are candidates for selection."""
        assert not np.all(self.y == self.pom_pad), \
            "cannot sample coordinates from empty map"
        # take a random sample of positive labels
        candidate_indices = np.argwhere(self.y == 1 - self.pom_pad)
        # apply the coordinate margin
        min_yx = np.array([coord_margin, coord_margin])
        max_yx = np.array(self.y.shape) - coord_margin
        candidate_indices = candidate_indices[
            np.all(candidate_indices < max_yx, axis=1) &
            np.all(candidate_indices > min_yx, axis=1)]
        print("[INFO] sampling {} training coordinates from {} candidate pixels".format(
            n, len(candidate_indices)
        ))
        sample_ix = np.random.choice(candidate_indices.shape[0], size=n, replace=False)
        sample = candidate_indices[sample_ix, :]
        sample = [xy.tolist() for xy in sample]
        return sample

    def generate_training_coordinates(self, out_dir, n, coord_margin):
        coords = self.sample_training_coordinates(n, coord_margin)
        self.write_training_coordiates(coords, out_dir)
