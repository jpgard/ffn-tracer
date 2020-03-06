import glob
import numpy as np
import os.path as osp
from typing import Optional

from skimage.io import imread
from ffn.utils import bounding_box

from fftracer.datasets import PairedDataset2d
from mozak.utils.connectors import ImageAPIConnector
from mozak.datasets.gold_standard import MozakGoldStandardTrace
from mozak.datasets.trace import nodes_and_edges_to_trace, nodes_and_edges_to_volume
from fftracer.utils import VALID_IMAGE_EXTENSIONS

# Thresholds to use for uniform sampling of training examples by fraction of active
# voxels, as implemented in original FFN paper (Januszewski et al 2016)
FA_THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.1,
                 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


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

    def load_x_data(self, data_dir=None):
        if data_dir is None:
            self.load_x_from_dap()
        else:
            self.load_x_from_file(data_dir)

    def load_y_data(self, gs_dir):
        gs = MozakGoldStandardTrace(self.dataset_id, gs_dir)
        gs.fetch_trace()
        # create "soft labels" map
        self.y = nodes_and_edges_to_trace(gs.nodes, gs.edges, imshape=self.x.shape,
                                          trace_value=self.label_value,
                                          pad_value=self.pom_pad)

    def load_data(self, gs_dir, data_dir=None):
        """
        Load the image data and the gold standard data.
        :param gs_dir: directory containing gold-standard trace (y).
        :param data_dir: directory containing the input image data (x).
        :return: None.
        """
        self.load_x_data(data_dir)
        # get the mask data for y
        self.load_y_data(gs_dir)
        self.check_xy_shapes_match()
        return

    def get_pos_label_locations(self, coord_margin: int) -> np.array:
        """Fetch the set of (y,x) locations for which the label is positive."""
        candidate_indices = np.argwhere(self.y == self.label_value)
        # apply the coordinate margin
        min_yx = np.array([coord_margin, coord_margin])
        max_yx = np.array(self.y.shape) - coord_margin
        candidate_indices = candidate_indices[
            np.all(candidate_indices < max_yx, axis=1) &
            np.all(candidate_indices > min_yx, axis=1)]
        return candidate_indices

    def uniformly_sample_training_coordinates(self, n: int, coord_margin: int):
        """Create a list of (x,y) coordinates for training. Only coordinates with
        positive labels are candidates for selection."""
        candidate_indices = self.get_pos_label_locations(coord_margin)

        print("[INFO] uniformly sampling n = {} training coordinates from {} candidate "
              "pixels".format(n, len(candidate_indices))
              )
        sample_ix = np.random.choice(candidate_indices.shape[0], size=n, replace=False)
        sample = candidate_indices[sample_ix, :]
        sample = [xy.tolist() for xy in sample]
        return sample

    def proportionally_sample_training_coordinates(self, p: float, coord_margin: int):
        candidate_indices = self.get_pos_label_locations(coord_margin)
        n = int(p * candidate_indices.shape[0])
        print("[INFO] sampling {} training coordinates from {} candidate pixels; p={}"
              .format(n, len(candidate_indices), p)
              )
        sample_ix = np.random.choice(candidate_indices.shape[0], size=n, replace=False)
        sample = candidate_indices[sample_ix, :]
        sample = [xy.tolist() for xy in sample]
        return sample


    def generate_training_coordinates(self, out_dir, n=None, coord_margin=0,
                                      method: Optional[str] = None,
                                      coord_sampling_prob: Optional[float] = None):
        assert not np.all(self.y == self.pom_pad), \
            "cannot generate training coordinates from empty labels map"
        if method == "uniform_by_dataset":
            coords = self.uniformly_sample_training_coordinates(n, coord_margin)
        if method == "proportional_by_dataset":
            coords = self.proportionally_sample_training_coordinates(
                coord_sampling_prob, coord_margin)
        else:
            raise NotImplementedError
        return coords

    def generate_and_write_training_coordinates(
            self, out_dir, **kwargs):
        coords = self.generate_training_coordinates(
            out_dir, **kwargs)
        self._write_training_coordinates(coords, out_dir)


class MozakDataset3d(MozakDataset2d):
    """A container for 3d mozak data."""

    def load_x_from_dap(self):
        from mozak.datasets.img import MozakNeuronVolume
        # TODO(jpgard): find the res_level corresponding to the same resolution as the
        #  image pixels. This is probably res_level=0, but note that this will be slow
        #  to load.
        import ipdb;
        ipdb.set_trace()
        volume = MozakNeuronVolume(self.dataset_id, res_level=4)
        volume.fetch_image()
        self.x = volume.img

    def load_data(self, gs_dir, data_dir=None):
        """
        Load the image data and the gold standard data.
        :param gs_dir: directory containing gold-standard trace.
        :return: None.
        """
        if not data_dir:
            self.load_x_from_dap()
        else:
            # TODO(jpgard): this should load a 3D file by calling
            #  MozakNeuronVolume.load_npy();
            #  currently these are not supported.
            raise NotImplementedError

        # get the mask data for y
        gs = MozakGoldStandardTrace(self.dataset_id, gs_dir)
        gs.fetch_trace()
        # create "soft labels" map
        self.y = nodes_and_edges_to_volume(gs.nodes, gs.edges,
                                           # TODO(jpgard): set all dimensions from 3d data
                                           imshape=[100,
                                                    self.x.shape[1],
                                                    self.x.shape[0]],
                                           trace_value=self.label_value,
                                           # TODO(jpgard): migrate code to use 0-1
                                           #  arrays, not pre-softened arrays.
                                           pad_value=self.pom_pad
                                           )
        import ipdb;
        ipdb.set_trace()
        # TODO(jpgard): write the Y volume to slices and then convert to HDF5; then,
        #  check the results in a viewer to see whether they correspond to the neural
        #  image and tracing is correct(probably will require some reflection due to
        #  shifting of axes, at the least).
        self.check_xy_shapes_match()
        return
