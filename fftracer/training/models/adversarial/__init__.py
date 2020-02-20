class Discriminator:
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
