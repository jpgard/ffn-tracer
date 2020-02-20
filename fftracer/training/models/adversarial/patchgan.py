"""
Class to emulate the PatchGAN discriminator network.

See https://arxiv.org/pdf/1611.07004.pdf and
https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_discriminator .
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
        assert self.dim == 2
        net = drop_axis(net, axis=1, name="drop_z_2d_discriminator")
        batch_size = net.get_shape().as_list()[0]
        input_batch_shape = net.get_shape().as_list()[1:]
        self.check_valid_input_shape(input_batch_shape)
        kw = (4, 4)  # kernel size of PatchGAN
        with tf.variable_scope(self.d_scope_name, reuse=False):
            h0 = tf.keras.layers.Conv2D(df_dim, kw, strides=(2, 2),
                                        padding='SAME',
                                        input_shape=self.input_shape,
                                        activation=tf.nn.leaky_relu)(net)
            h0 = tf.keras.layers.BatchNormalization()(h0)
            # h0 has shape [FOV/2, FOV/2, self.df_dim]
            h1 = tf.keras.layers.Conv2D(df_dim * 2, kw, strides=(2, 2),
                                        padding='SAME',
                                        activation=tf.nn.leaky_relu)(h0)
            h1 = tf.keras.layers.BatchNormalization()(h1)
            # h1 has shape [FOV/4, FOV/4, self.df_dim*2]
            h2 = tf.keras.layers.Conv2D(df_dim * 4, kw, strides=(2, 2),
                                        padding='SAME',
                                        activation=tf.nn.leaky_relu)(h1)
            h2 = tf.keras.layers.BatchNormalization()(h2)
            # h2 has shape [FOV/8, FOV/8, self.df_dim*4]
            h3 = tf.keras.layers.Conv2D(df_dim * 8, kw, strides=(1, 1),
                                        padding='SAME')(h2)
            # h3 has shape [FOV/16, FOV/16, self.df_dim*8]
            # Note that no batch normalization is applied at final layer and linear
            # activation is used.

            # The model is parameterized such that the effective receptive field of
            # each output of the network maps to a specific size in the input image.
            # The output of the network is a single feature map of real/fake
            # predictions that can be averaged to give a single score.
            # See https://machinelearningmastery.com/a-gentle-introduction-to-pix2pix
            # -generative-adversarial-network/
            outputs = tf.reshape(h3, [batch_size, -1])
            outputs = tf.math.reduce_sum(outputs, axis=0)
            # outputs should have shape [batch_size,]
            assert tf.get_shape.as_list(outputs) == [batch_size, ]
            return outputs

    def predict_discriminator(self, batch):
        if self.dim == 2:
            return self.predict_discriminator_2d(batch)
        elif self.dim == 3:
            raise NotImplementedError

    def discriminator_loss(self, real_output, fake_output):
        pass

