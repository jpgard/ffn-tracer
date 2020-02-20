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

    def predict_discriminator(self, batch):
        """
        Use a DCGAN-style discriminator to predict whether a batch is real or fake.

        The discriminator takes a tensor of shape [batch_size, y, x, num_channels] and
        outputs a single real value indicating whether this is a batch of real or fake
        samples.

        :param batch: the batch to predict on.
        :return: a single-element Tensor with the predicted probability that the batch
        is real.
        """
        if self.dim == 2:
            return self.predict_discriminator_2d(batch)
        elif self.dim == 3:
            raise NotImplementedError

    def get_real_labels(self, real_output: tf.Tensor) -> tf.Tensor:
        """Get a (possibly smoothed/noisy) set of labels for real_output."""
        if self.smooth_labels:
            noisy_labels = tf.random.normal(shape=real_output.get_shape(),
                                            mean=self.noisy_label_mean,
                                            stddev=self.noisy_label_stddev)
            return noisy_labels
        else:
            return tf.ones_like(real_output)

    def discriminator_loss(self, real_output, fake_output):
        """Compute the loss for a discriminator and save as self.d_loss ."""
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # create labels and add noise to the real labels if desired
        real_labels = self.get_real_labels(real_output)
        fake_labels = tf.zeros_like(fake_output)

        real_loss = cross_entropy(real_labels, real_output)
        fake_loss = cross_entropy(fake_labels, fake_output)
        discriminator_loss_batch = real_loss + fake_loss

        discriminator_loss_batch = tf.verify_tensor_all_finite(
            discriminator_loss_batch, 'Invalid discriminator loss')
        self.d_loss = tf.reduce_mean(discriminator_loss_batch)
        tf.summary.scalar('discriminator_loss', self.d_loss)
        return

    def get_optimizer(self):
        if self.optimizer_name == "adam":
            # Use the default values from DCGAN paper; they said lower learning rate and
            # beta_1 necessary to improve stability
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        elif self.optimizer_name == "sgd":
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
