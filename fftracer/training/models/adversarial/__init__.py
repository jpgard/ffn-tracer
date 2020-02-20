from abc import ABC, abstractmethod
import tensorflow as tf

class Discriminator(ABC):
    """Parent class that adversarial losses should inherit from."""

    def __init__(self, input_shape, optimizer_name: str, smooth_labels: bool, dim=2,
                 d_scope_name='dcgan_discriminator',
                 noisy_label_mean=0.9, noisy_label_stddev=0.025, learning_rate=0.0001):
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

    @abstractmethod
    def predict_discriminator(self, batch):
        """Make a prediction for the discriminator on a batch of examples.

        The discriminator takes a tensor of shape [batch_size, y, x, num_channels] and
        outputs a single real value indicating whether this is a batch of real or fake
        samples.
        """
        raise

    @abstractmethod
    def discriminator_loss(self, real_output, fake_output):
        """Compute the loss for a discriminator and save as self.d_loss ."""
        raise

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
