"""Classes for FFN model definition."""

import numpy as np

from scipy.ndimage import distance_transform_edt as distance

from ffn.training.model import FFNModel
import tensorflow as tf


def _predict_object_mask_2d(net, depth=9):
    """Computes single-object mask prediction for 2d using a 3d conv with 3x3x1 kernel.

    Modified from ffn.training.models.convstack_3d .
    """
    conv = tf.contrib.layers.conv3d

    with tf.contrib.framework.arg_scope([conv], num_outputs=32,
                                        kernel_size=(3, 3, 1),
                                        padding='SAME'):
        net = conv(net, scope='conv0_a')
        net = conv(net, scope='conv0_b', activation_fn=None)
        for i in range(1, depth):
            with tf.name_scope('residual%d' % i):
                in_net = net
                net = tf.nn.relu(net)
                net = conv(net, scope='conv%d_a' % i)
                net = conv(net, scope='conv%d_b' % i, activation_fn=None)
                net += in_net
    net = tf.nn.relu(net)
    logits = conv(net, 1, (1, 1, 1), activation_fn=None, scope='conv_lom')

    return logits


class FFNTracerModel(FFNModel):
    """Base class for FFN tracing models."""

    def __init__(self, deltas, batch_size=None, dim=2,
                 fov_size=None, depth=9, loss_name="sigmoid_pixelwise", alpha=1e-6):
        """

        :param deltas:
        :param batch_size:
        :param dim: number of dimensions of model prediction (e.g. 2 = 2D input/output)
        :param fov_size: [x,y,z] fov size.
        :param depth: number of convolutional layers.
        """
        self.dim = dim
        assert (0 < alpha < 1), "alpha must be in range (0,1)"
        super(FFNTracerModel, self).__init__(deltas, batch_size)

        self.deltas = deltas
        self.batch_size = batch_size
        self.depth = depth
        self.loss_name = loss_name
        self.alpha = alpha
        # The seed is always a placeholder which is fed externally from the
        # training/inference drivers.
        self.input_seed = tf.placeholder(tf.float32, name='seed')
        self.input_patches = tf.placeholder(tf.float32, name='patches')

        # Set pred_mask_size = input_seed_size = input_image_size = fov_size and
        # also set input_seed.shape = input_patch.shape = [batch_size, z, y, x, 1] .
        self.set_uniform_io_size(fov_size)

    def alpha_weight_losses(self, loss_a, loss_b):
        """Compute alpha * loss_a + (1 - alpha) loss_b and set to self.loss.

        Computes the scheduled alpha, then apply it to compute the total weighted
        loss. The alpha scheduling this is a hockey-stick shaped decay where the
        contribution of the ce_loss bottoms out after reaching 0.01. This happens in
        (1 - 0.01)/alpha = 990,000 epochs (using a min alpha of 0.01 and alpha = 1e-6).
        """

        alpha = tf.maximum(
            1. - self.alpha * tf.cast(self.global_step, tf.float32),
            0.01
        )
        self.loss = (alpha * loss_a) + (1. - alpha) * loss_b
        tf.summary.scalar("alpha_loss", self.loss)
        self.loss = tf.verify_tensor_all_finite(self.loss, 'Invalid loss detected')

    def set_up_l1_loss(self, logits):
        """Set up l1 loss."""
        assert self.labels is not None
        assert self.loss_weights is not None

        pixel_loss = tf.abs(self.labels - logits)
        pixel_loss *= self.loss_weights
        self.loss = tf.reduce_mean(pixel_loss)
        tf.summary.scalar('l1_loss', self.loss)
        self.loss = tf.verify_tensor_all_finite(self.loss, 'Invalid loss detected')
        return

    def set_up_ssim_loss(self, logits):
        """Set up structural similarity index (SSIM) loss.

        SSIM loss does not support per-pixel weighting.
        """
        assert self.labels is not None

        image_loss = tf.image.ssim(self.labels, logits, max_val=1.0)

        # High values of SSIM indicate good quality, but the model will minimize loss,
        # so we reverse the sign of loss.
        image_loss = tf.math.negative(image_loss)

        self.loss = tf.reduce_mean(image_loss)
        tf.summary.scalar('ssim_loss', self.loss)
        self.loss = tf.verify_tensor_all_finite(self.loss, 'Invalid loss detected')
        return

    def set_up_ms_ssim_loss(self, logits):
        """Set up multiscale structural similarity index (MS-SSIM) loss.

        MS-SSIM loss does not support per-pixel weighting.
        """
        assert self.labels is not None

        # Compute the MS-SSIM; use a filter size of 4 because this is the largest
        # filter that can run over the data with FOV = [1,49,49] without raising an
        # error due to insufficient input size (note that default filter_size=11).

        image_loss = tf.image.ssim_multiscale(self.labels, logits, max_val=1.0,
                                              # to use original values:
                                              # power_factors=(0.0448, 0.2856, 0.3001),
                                              power_factors=[float(1) / 3] * 3,
                                              filter_size=4)

        # High values of MS-SSIM indicate good quality, but the model will minimize loss,
        # so we reverse the sign of loss.
        image_loss = tf.math.negative(image_loss)

        self.loss = tf.reduce_mean(image_loss)
        tf.summary.scalar('ms_ssim_loss', self.loss)
        self.loss = tf.verify_tensor_all_finite(self.loss, 'Invalid loss detected')
        return

    def set_up_boundary_loss(self, logits):
        """Based on 'Boundary Loss for Highly Unbalanced Segmentation', Kervadec et al.

        Code based on initial implementation at link within the official repo:
        https://github.com/LIVIAETS/surface-loss/issues/14#issuecomment-546342163
        """
        assert self.labels is not None
        assert self.loss_weights is not None

        def calc_dist_map(seg):
            """Calculate the distance map for a ground truth segmentation."""
            res = np.zeros_like(seg)
            # Form a boolean mask from "soft" labels, which are set to 0.95 for FFN.
            posmask = (seg >= 0.95).astype(np.bool)

            if posmask.any():
                negmask = ~posmask
                res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

            return res

        def calc_dist_map_batch(y_true):
            """Calculate the distance map for the batch."""
            return np.array([calc_dist_map(y)
                             for y in y_true]).astype(np.float32)

        # Compute the boundary loss
        y_true_dist_map = tf.py_func(func=calc_dist_map_batch,
                                         inp=[self.labels],
                                         Tout=tf.float32)
        boundary_loss = tf.math.multiply(logits, y_true_dist_map, "SurfaceLoss")
        batch_boundary_loss = tf.reduce_mean(boundary_loss)
        tf.summary.scalar('boundary_loss', batch_boundary_loss)

        # Compute the pixel-wise cross entropy loss
        pixel_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                labels=self.labels)
        pixel_ce_loss *= self.loss_weights
        batch_ce_loss = tf.reduce_mean(pixel_ce_loss)
        tf.summary.scalar('pixel_loss', batch_ce_loss)

        self.alpha_weight_losses(batch_ce_loss, batch_boundary_loss)



    def set_up_loss(self, logit_seed):
        """Set up the loss function of the model."""
        if self.loss_name == "sigmoid_pixelwise":
            self.set_up_sigmoid_pixelwise_loss(logit_seed)
        elif self.loss_name == "l1":
            self.set_up_l1_loss(logit_seed)
        elif self.loss_name == "ssim":
            self.set_up_ssim_loss(logit_seed)
        elif self.loss_name == "ms_ssim":
            self.set_up_ms_ssim_loss(logit_seed)
        elif self.loss_name == "boundary":
            self.set_up_boundary_loss(logit_seed)
        else:
            raise NotImplementedError

    def define_tf_graph(self):
        """Modified for 2D from ffn.training.models.convstack_3d.ConvStack3DFFNModel ."""
        self.show_center_slice(self.input_seed)
        if self.input_patches is None:
            self.input_patches = tf.placeholder(  # [batch_size, x, y, z, num_channels]
                tf.float32, [1] + list(self.input_image_size[::-1]) + [1],
                name='patches')

        # JG: concatenate the input patches and the current mask for input to the model
        net = tf.concat([self.input_patches, self.input_seed], 4)

        with tf.variable_scope('seed_update', reuse=False):
            logit_update = _predict_object_mask_2d(net, self.depth)
        logit_seed = self.update_seed(self.input_seed, logit_update)

        # Make predictions available, both as probabilities and logits.
        self.logits = logit_seed
        self.logistic = tf.sigmoid(logit_seed)

        if self.labels is not None:
            self.set_up_loss(logit_seed)
            self.set_up_optimizer()
            self.show_center_slice(logit_seed)
            self.show_center_slice(self.labels, sigmoid=False)
            self.add_summaries()

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
