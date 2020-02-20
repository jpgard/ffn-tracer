"""
Class to emulate the PatchGAN discriminator network.

See https://arxiv.org/pdf/1611.07004.pdf and
https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_discriminator .
"""

import tensorflow as tf

from fftracer.training.models.adversarial import Discriminator

class PatchGAN(Discriminator):
    def predict_discriminator_2d(self, batch):
        pass
    def predict_discriminator(self, batch):
        pass