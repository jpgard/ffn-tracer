import tensorflow as tf

from fftracer.utils.tensor_ops import drop_axis


class DCGAN:
    def __init__(self, input_shape, optimizer_name: str, smooth_labels: bool, dim=2,
                 noisy_label_mean=0.9, noisy_label_stddev=0.025):
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
        self.d_scope_name = 'dcgan_discriminator'
        self.optimizer_name = optimizer_name
        self.smooth_labels = smooth_labels
        self.noisy_label_mean = noisy_label_mean
        self.noisy_label_stddev = noisy_label_stddev

    def predict_discriminator_2d(self, net):
        """
        Use a DCGAN-style discriminator to predict whether a batch is real or fake.

        The discriminator takes a tensor of shape [batch_size, y, x, num_channels] and
        outputs a single real value indicating whether this is a batch of real or fake
        samples.
        """
        if self.dim == 2:
            net = drop_axis(net, axis=1, name="drop_z_2d_discriminator")
        input_batch_shape = net.get_shape().as_list()[1:]
        assert input_batch_shape == self.input_shape, \
            "discriminator input has shape {}, does not match expected shape {}".format(
                 input_batch_shape, self.input_shape
            )
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
        """Compute the loss for a discriminator.

        returns a Tensor of shape [batch_size, 1].
        """
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
            return tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        elif self.optimizer_name == "sgd":
            return tf.train.GradientDescentOptimizer(learning_rate=0.0001)
