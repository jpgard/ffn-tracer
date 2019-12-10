"""Classes for FFN model definition."""

from ffn.training.model import FFNModel
import tensorflow as tf


class FFNTracerModel(FFNModel):
    """Base class for FFN tracing models."""

    def __init__(self, deltas, batch_size=None, dim=2,
                 fov_size=None):
        """

        :param deltas:
        :param batch_size:
        :param define_global_step:
        :param dim: number of dimensions of model prediction (e.g. 2 = 2D input/output)
        :param fov_size: [x,y,z] fov size.
        """
        self.dim = dim
        self.deltas = deltas
        self.batch_size = batch_size
        self.input_seed_size = None
        # The seed is always a placeholder which is fed externally from the
        # training/inference drivers.
        self.input_seed = tf.compat.v1.placeholder(tf.float32, name='seed')
        self.input_patches = tf.compat.v1.placeholder(tf.float32, name='patches')

        if fov_size:
            self.set_uniform_io_size(fov_size)
