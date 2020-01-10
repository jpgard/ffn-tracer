"""Seed policies for inference."""

import numpy as np

from ffn.inference.seed import BaseSeedPolicy


class ManualSeedPolicy(BaseSeedPolicy):
    """Use a manually-specified set of seeds.

    This class currently needs to be inserted in ffn.inference.seed.py to work properly.
    """
    def __init__(self, canvas, **kwargs):
      logging.info("ManualSeedPolicy.__init__()")
      super(ManualSeedPolicy, self).__init__(canvas, **kwargs)

    def _init_coords(self):
        # TODO(jpgard): collect these from user; temporarily these are hard-coded.
        coords = [(0, 4521, 3817), ]
        logging.info('ManualSeedPolicy: starting with coords {}'.format(coords))
        self.coords = np.array(coords)

    def __next__(self):
        """Returns the next seed point as (z, y, x).

        Does initial filtering of seed points to exclude locations that are
        too close to the image border.

        Returns:
          (z, y, x) tuples.

        Raises:
          StopIteration when the seeds are exhausted.
        """
        if self.coords is None:
            self._init_coords()

        while self.idx < self.coords.shape[0]:
            curr = self.coords[self.idx, :]
            self.idx += 1
            return tuple(curr)  # z, y, x

        raise StopIteration()
