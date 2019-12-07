"""Classes for FFN model definition."""

from ffn.training.model import FFNModel


class FFNTracerModel(FFNModel):
    """Base class for FFN tracing models."""

    def __init__(self, deltas, batch_size=None, define_global_step=True, dim=2):
        self.dim = dim
        self.deltas = deltas
        self.batch_size = batch_size
