import tensorflow as tf

from fftracer.utils.tensor_ops import drop_axis


class DCGAN:
    def __init__(self, input_shape, dim=2):
        assert dim in (2, 3)
        assert len(input_shape) == dim + 1, "input should have shape (dim + 1)"
        self.dim = dim
        self.input_shape = input_shape

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
        with tf.variable_scope('discriminator', reuse=False):
            net = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='SAME',
                                         input_shape=self.input_shape)(net)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
            net = tf.keras.layers.Dropout(0.3)(net)
            net = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(net)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
            net = tf.keras.layers.Dropout(0.3)(net)
            net = tf.keras.layers.Flatten()(net)
            batch_pred = tf.keras.layers.Dense(1)(net)
            # TODO(jpgard): should we apply sigmoid here, since we want a value in
            #  range (0,1) ?
            return batch_pred

    def predict_discriminator(self, batch):
        if self.dim == 2:
            return self.predict_discriminator_2d(batch)
        elif self.dim == 3:
            raise NotImplementedError
