import cv2

import numpy as np
import os.path as osp

from fftracer.datasets import PairedDataset2d, Seed
from collections import namedtuple
from mozak.utils.connectors import ImageAPIConnector
from mozak.datasets.gold_standard import MozakGoldStandardTrace
from mozak.datasets.trace import nodes_and_edges_to_trace


class MozakDataset2d(PairedDataset2d):

    def load_data(self, gs_dir):
        """
        Load the image data and the gold standard data.
        :param gs_dir: directory containing gold-standard trace.
        :return: None.
        """
        # TODO(jpgard): request an update to the API so center and size can be left
        #  unspecified to fetch the entire image, instead of specifying here
        imshape = (76 * 10 ** 2, 76 * 10 ** 2)
        center = list(map(lambda x: x // 2, imshape))
        # get the image data for x
        img_api = ImageAPIConnector()
        img_bytes = img_api.request_image(self.dataset_id, center=center, size=imshape)
        self.x = np.asarray(img_bytes.convert("L"))
        # get the mask data for y
        gs = MozakGoldStandardTrace(self.dataset_id, gs_dir)
        gs.fetch_trace()
        # create "soft labels" map
        self.y = nodes_and_edges_to_trace(gs.nodes, gs.edges, imshape,
                                          trace_value=0.95, pad_value=0.05)
        self.check_xy_shapes_match()
        return

