"""
Class to emulate the PatchGAN discriminator network.

See https://arxiv.org/pdf/1611.07004.pdf and
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob
/8cda06f7c36b012769efac63adc1a68586b8fb85/models/networks.py#L538
"""

import tensorflow as tf

from fftracer.training.models.adversarial import Discriminator
from fftracer.utils.tensor_ops import drop_axis


class PatchGAN(Discriminator):
    def predict_discriminator_2d(self, net):
        """
        The PatchGAN discriminator.

        Adapted from the TensorFlow implementation of pix2pix linked in the pix2pix repo,
        here: https://github.com/yenchenlin/pix2pix-tensorflow .
        :param batch:
        :return:
        """
        df_dim = 64
        kernel = 3
        stride = 2
        assert self.dim == 2
        net = drop_axis(net, axis=1, name="drop_z_2d_discriminator")
        batch_size = net.get_shape().as_list()[0]
        input_batch_shape = net.get_shape().as_list()[1:]
        self.check_valid_input_shape(input_batch_shape)
        with tf.variable_scope(self.d_scope_name, reuse=False):
            net = tf.keras.layers.Conv2D(df_dim, kernel, strides=stride,
                                         padding='SAME',
                                         input_shape=self.input_shape,
                                         activation=tf.nn.leaky_relu)(net)
            net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.Conv2D(df_dim * 2, kernel, strides=stride,
                                         padding='SAME',
                                         activation=tf.nn.leaky_relu)(net)

            # net = tf.keras.layers.Conv2D(df_dim * 8, kernel, strides=(1, 1),
            #                             padding='SAME')(net)

            # Note that no batch normalization is applied at final layer and linear
            # activation is used.

            # The model is parameterized such that the effective receptive field of
            # each output of the network maps to a specific size in the input image.
            # The output of the network is a single feature map of real/fake
            # predictions that can be averaged to give a single score.
            # See https://machinelearningmastery.com/a-gentle-introduction-to-pix2pix
            # -generative-adversarial-network/

            outputs = tf.reshape(net, [batch_size, -1])
            outputs = tf.math.reduce_sum(outputs, axis=1)
            # outputs should have shape [batch_size,]
            assert outputs.get_shape().as_list() == [batch_size, ]
            return outputs

    def predict_discriminator(self, batch):
        if self.dim == 2:
            return self.predict_discriminator_2d(batch)
        elif self.dim == 3:
            raise NotImplementedError
