"""
Class to emulate the DCGAN discriminator network.
"""

import tensorflow as tf

from fftracer.utils.tensor_ops import drop_axis
from fftracer.training.models.adversarial import Discriminator


class DCGAN(Discriminator):

    def predict_discriminator_2d(self, net):
        """
        Use a DCGAN-style discriminator to predict whether a batch is real or fake.

        The discriminator takes a tensor of shape [batch_size, y, x, num_channels] and
        outputs a single real value indicating whether this is a batch of real or fake
        samples.
        """
        assert self.dim == 2
        net = drop_axis(net, axis=1, name="drop_z_2d_discriminator")
        input_batch_shape = net.get_shape().as_list()[1:]
        self.check_valid_input_shape(input_batch_shape)
        with tf.variable_scope(self.d_scope_name, reuse=False):
            net = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='SAME',
                                         input_shape=self.input_shape)(net)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
            net = tf.keras.layers.Dropout(0.3)(net)
            net = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(net)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
            net = tf.keras.layers.Dropout(0.3)(net)
            net = tf.keras.layers.Flatten()(net)
            batch_pred = tf.keras.layers.Dense(1)(net)
            return batch_pred

