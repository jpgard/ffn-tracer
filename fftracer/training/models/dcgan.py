import tensorflow as tf
from tensorflow.keras import layers


def make_discriminator_model(input_shape):
    """
    Make a DCGAN-style discriminator.

    The discriminator takes a tensor of shape [batch_size, z, y, x, num_channels] and
    outputs a single real value indicating whether this is a batch of real or fake
    samples.
    :return:
    """
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
