from abc import ABC, abstractmethod

import tensorflow as tf


class Discriminator(ABC):
    """Parent class that adversarial losses should inherit from.

    In addition to implementing a subclass, at least predict_discriminator_2d() must be
    implemented for any subclass of Discriminator.

    """

    def __init__(self, input_shape, optimizer_name: str, smooth_labels: bool, dim=2,
                 d_scope_name='discriminator',
                 noisy_label_mean=0.9, noisy_label_stddev=0.025, learning_rate=0.0001,
                 spectral_normalization=False):
        """

        :param input_shape: the shape of the input images, omitting batch size.
        :param optimizer_name: name of the optimizer to use.
        :param smooth_labels: boolean indicator for whether to perform one-sided label
        smoothing in the discriminator (see https://arxiv.org/pdf/1701.00160.pdf).
        :param dim: the dimension of input images.
        """
        assert dim in (2, 3)
        assert len(input_shape) == dim + 1, "input should have shape (dim + 1)"
        self.dim = dim
        self.input_shape = input_shape
        self.d_loss = None  # Placeholder for the discriminator loss, defined below.
        self.d_scope_name = d_scope_name
        self.optimizer_name = optimizer_name
        self.smooth_labels = smooth_labels
        self.noisy_label_mean = noisy_label_mean
        self.noisy_label_stddev = noisy_label_stddev
        self.learning_rate = learning_rate
        self.spectral_normalization = spectral_normalization

    @abstractmethod
    def predict_discriminator_2d(self, batch):
        raise

    @abstractmethod
    def get_conv(self):
        """Get the convolution operation."""
        raise

    def predict_discriminator(self, batch):
        """Make a prediction for the discriminator on a batch of examples.

        The discriminator takes a tensor of shape [batch_size, y, x, num_channels] and
        outputs a Tensor containing asingle real value indicating whether this is a
        batch of real or fake samples.
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

    def check_valid_input_shape(self, input_batch_shape):
        assert input_batch_shape == self.input_shape, \
            "discriminator input has shape {}, does not match expected shape {}".format(
                input_batch_shape, self.input_shape
            )
        return

    def get_optimizer(self):
        """Instantiate and return an optimizer for this discriminator model."""

        if self.optimizer_name == "adam":
            # Use the default values from DCGAN paper; they said lower learning rate and
            # beta_1 necessary to improve stability
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        elif self.optimizer_name == "sgd":
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
