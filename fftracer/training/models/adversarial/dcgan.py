"""
Class to emulate the DCGAN discriminator network.
"""

from functools import partial

import tensorflow as tf

from fftracer.utils.tensor_ops import drop_axis
from fftracer.training.models.adversarial import Discriminator


def spectral_norm(w, layer, iteration=1):
    """
    Compute the spectral norm of weight matrix w via the power iteration method.

    The equations reference the arXiv version of the ICLR 2018 paper, which can be
    accessed at https://arxiv.org/abs/1802.05957 .
    Code adapted from https://github.com/taki0112/Spectral_Normalization-Tensorflow .

    :param w: The weight matrix to normalize.
    :param iteration: Number of iterations of the power iteration step to perform. Note
    that 1 step is considered enough.
    :return: a Tensor of shape [filter_height, filter_width, in_channels, out_channels].
    """
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u_hat = tf.get_variable("u_hat/{}".format(layer), [1, w_shape[-1]],
                            initializer=tf.random_normal_initializer(),
                            trainable=False)
    # Initialize empty v_hat and run power iteration
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)  # This is Equation (20) of Algorithm 1 in paper
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)  # This is Equation (21) in paper
    # Stop gradients so u_hat and v_hat are not updated by the optimizer
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    # Approximate the spectral norm of W using the results of power iteration
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))  # Equation (19)
    # Normalize W and reshape to its original shape
    with tf.control_dependencies([sigma]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


class DCGAN(Discriminator):
    def get_conv(self, k=5, s=2, channels=1):
        """
        Get the convolution operation for this model.
        :param k: kernel size (uses same size for both dimensions by default).
        :param s: stride size (uses same size for both dimensions by default).
        :param channels: number of channels (should always be 1).
        :return: a callable implementing the desired convolution operation.
        """
        if not self.spectral_normalization:
            return partial(tf.keras.layers.Conv2D, kernel_size=(k, k), strides=(s, s),
                           padding='SAME', activation=tf.nn.leaky_relu)
        else:
            def conv_spectral_norm(net, filters, layer):
                in_channels = net.get_shape()[-1]
                weights = tf.get_variable("kernel/{}".format(layer),
                                          shape=[k, k, in_channels, filters])
                bias = tf.get_variable("bias/{}".format(layer), [filters],
                                       initializer=tf.constant_initializer(0.0))
                filter_tensor = spectral_norm(weights, layer)
                print("layer {} conv_spectral_norm(): filter tensor shape: {}".format(
                    layer, filter_tensor.get_shape().as_list()))
                return tf.nn.conv2d(input=net, filter=filter_tensor,
                                    strides=[s, s], padding='SAME') + bias

            return conv_spectral_norm

    def predict_basic_dcgan_2d(self, net):
        """Implements a basic DCGAN discriminator model."""
        conv = self.get_conv()

        with tf.variable_scope(self.d_scope_name, reuse=False):
            net = conv(filters=64, input_shape=self.input_shape)(net)
            # net is Tensor of shape [batch_size, k**2, k**2, 64]
            # (so [batch_size, 25, 25, 64] with kernel of size 5)
            net = tf.keras.layers.Dropout(0.3)(net)
            net = conv(filters=128)(net)
            # net is Tensor of shape [batch_size, k**2 // 2 + 1, k**2 // 2 + 1, 64]
            # (so [batch_size, 13, 13, 128] with kernel of size 5)
            net = tf.keras.layers.Dropout(0.3)(net)
            net = tf.keras.layers.Flatten()(net)
            batch_pred = tf.keras.layers.Dense(1)(net)
            return batch_pred

    def predict_spectralnorm_dcgan_2d(self, net):
        """Implements a DCGAN discriminator model with spectral normalization"""
        conv = self.get_conv()
        with tf.variable_scope(self.d_scope_name, reuse=tf.AUTO_REUSE):
            # net is a Tensor of shape [batch_size, y, x, num_channels]
            net = conv(net, filters=64, layer=1)
            assert net.get_shape().as_list() == [4, 25, 25, 64]
            net = tf.keras.layers.Dropout(0.3)(net)
            net = conv(net, filters=128, layer=2)
            assert net.get_shape().as_list() == [4, 13, 13, 128]
            net = tf.keras.layers.Dropout(0.3)(net)
            net = tf.keras.layers.Flatten()(net)
            batch_pred = tf.keras.layers.Dense(1)(net)
            return batch_pred

    def predict_discriminator_2d(self, net):
        """
        Use a DCGAN-style discriminator to predict whether a batch is real or fake.

        The discriminator takes a tensor of shape [batch_size, y, x, num_channels] and
        outputs a single real value indicating whether this is a batch of real or fake
        samples.
        """
        # check the inputs and drop the z-axis
        assert self.dim == 2
        net = drop_axis(net, axis=1, name="drop_z_2d_discriminator")
        # net is a Tensor of shape [batch_size, y, x, num_channels]
        input_batch_shape = net.get_shape().as_list()[1:]
        self.check_valid_input_shape(input_batch_shape)

        if not self.spectral_normalization:
            return self.predict_basic_dcgan_2d(net)
        else:
            # use spectral normalization
            return self.predict_spectralnorm_dcgan_2d(net)
