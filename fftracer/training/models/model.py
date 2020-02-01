"""Classes for FFN model definition."""

import logging
import numpy as np
import math

from scipy.ndimage import distance_transform_edt as distance

from ffn.training.model import FFNModel
from fftracer.training.self_attention.non_local import sn_non_local_block_sim
from fftracer.training.models.dcgan import DCGAN
import tensorflow as tf
from fftracer.utils.tensor_ops import drop_axis, add_axis


def _predict_object_mask_2d(net, depth=9, self_attention_index=None):
    """
    Computes single-object mask prediction for 2d.

    Modified from ffn.training.models.convstack_3d.
    :param net: the network input; for FFN this is a concatenation of the input image
    patch and the current POM.
    :param depth: number of residual blocks to use.
    :param self_attention_index: use a self-attention block instead of a normal
    residual block at this layer, if specified.
    :return: the model logits corresponding to the updated POM.
    """
    if self_attention_index:
        assert self_attention_index <= depth
    conv = tf.contrib.layers.conv3d

    with tf.contrib.framework.arg_scope([conv], num_outputs=32,
                                        kernel_size=(3, 3, 1),
                                        padding='SAME'):
        net = conv(net, scope='conv0_a')
        net = conv(net, scope='conv0_b', activation_fn=None)
        for i in range(1, depth):
            with tf.name_scope('residual%d' % i):
                # At each iteration, net has shape [batch_size, 1, y, x, num_outputs]
                if i == self_attention_index:
                    # Use a self-attention block instead of a residual block.

                    # Self-Attention only implemented for 2D; drop the z-axis and
                    # reconstruct it after the self-attention block.
                    net = drop_axis(net, axis=1)
                    net = sn_non_local_block_sim(net, None, "self_attention")
                    net = add_axis(net, axis=1)
                else:
                    # Use a residual block.
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

    def __init__(self, deltas, batch_size=None, dim=3,
                 fov_size=None, depth=9, loss_name="sigmoid_pixelwise", alpha=1e-6,
                 l1lambda=1e-3, self_attention_layer=None):
        """

        :param deltas:
        :param batch_size:
        :param dim: number of dimensions of model prediction (e.g. 2 = 2D input/output)
        :param fov_size: [x,y,z] fov size.
        :param depth: number of convolutional layers.
        """
        try:
            fov_size = [int(x) for x in fov_size]
            alpha = float(alpha)
        except Exception as e:
            logging.error("error parsing FFNTracerModel argument: {}".format(e))

        self.dim = dim
        assert (0 < alpha < 1), "alpha must be in range (0,1)"
        super(FFNTracerModel, self).__init__(deltas, batch_size)

        self.deltas = deltas
        self.batch_size = batch_size
        self.depth = depth
        self.loss_name = loss_name
        self.alpha = alpha
        self.fov_size = fov_size
        self.l1lambda = l1lambda
        self.self_attention_layer = self_attention_layer
        self.discriminator = None
        self.discriminator_loss = None
        # The seed is always a placeholder which is fed externally from the
        # training/inference drivers.
        self.input_seed = tf.placeholder(tf.float32, name='seed')
        self.input_patches = tf.placeholder(tf.float32, name='patches')

        # Set pred_mask_size = input_seed_size = input_image_size = fov_size and
        # also set input_seed.shape = input_patch.shape = [batch_size, z, y, x, 1] .
        self.set_uniform_io_size(fov_size)

    def compute_sce_loss(self, logits):
        """Compute the pixelwise sigmoid cross-entropy loss using logits and labels."""
        assert self.labels is not None
        assert self.loss_weights is not None
        pixel_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                labels=self.labels)
        pixel_ce_loss *= self.loss_weights
        batch_ce_loss = tf.reduce_mean(pixel_ce_loss)
        return batch_ce_loss

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

        ssim_loss = tf.image.ssim(self.labels, logits, max_val=1.0)

        # High values of SSIM indicate good quality, but the model will minimize loss,
        # so we reverse the sign of loss.
        ssim_loss = tf.math.negative(ssim_loss)

        batch_ssim_loss = tf.reduce_mean(ssim_loss)
        tf.summary.scalar('ssim_loss', batch_ssim_loss)

        # Compute the pixel-wise cross entropy loss
        batch_ce_loss = self.compute_sce_loss(logits)
        tf.summary.scalar('pixel_loss', batch_ce_loss)

        self.alpha_weight_losses(batch_ce_loss, batch_ssim_loss)

        self.loss = tf.verify_tensor_all_finite(self.loss, 'Invalid loss detected')
        return

    def set_up_ms_ssim_loss(self, logits):
        """Set up multiscale structural similarity index (MS-SSIM) loss.

        MS-SSIM loss does not support per-pixel weighting.
        """
        # TODO(jpgard): try updating this to use https://github.com/andrewekhalel/sewar
        #  imlpementation of ssim instead of tf.image version; this currently leads to
        #  some kind of error.

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
        # Compute the maximum euclidean distance for the model FOV size; this is used
        # to normalize the boundary loss and constrain it to the range (0,1) so it does
        # not dominate the loss function (otherwise boundary loss can take extreme
        # values, particularly as image size grows).
        max_dist = math.sqrt((self.fov_size[0] - 1)**2 +
                             (self.fov_size[1] - 1)**2 +
                             (self.fov_size[2] - 1)**2)

        def calc_dist_map(seg):
            """Calculate the distance map for a ground truth segmentation."""
            # Form a boolean mask from "soft" labels, which are set to 0.95 for FFN.
            posmask = (seg >= 0.95).astype(np.bool)
            assert posmask.any(), "ground truth must contain at least one active voxel"
            negmask = ~posmask
            res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            res /= max_dist
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
        batch_ce_loss = self.compute_sce_loss(logits)

        tf.summary.scalar('pixel_loss', batch_ce_loss)

        self.alpha_weight_losses(batch_ce_loss, batch_boundary_loss)

    def set_up_l1_continuity_loss(self, logits):
        """Sets up the l1 continuity loss.

        L1 continuity loss uses the normal cross-entropy loss with a regularizer which
        enforces 'contnuity' between pixels.
        """
        # Compute the pixel-wise cross entropy loss
        batch_ce_loss = self.compute_sce_loss(logits)
        tf.summary.scalar('pixel_loss', batch_ce_loss)
        row_wise_logits = tf.reshape(logits, [-1], 'FlattenRowWise')
        column_wise_logits = tf.reshape(tf.transpose(logits), [-1], 'FlattenColWise')
        # Compute the l1 continuity loss row-wise, subtracting each element from the
        # next element row-wise
        row_loss = row_wise_logits - tf.concat([row_wise_logits[1:], [0,]], 0)
        row_loss = tf.abs(row_loss)
        # Compute the l1 continuity loss column-wise.
        column_loss = column_wise_logits - tf.concat([column_wise_logits[1:], [0, ]], 0)
        column_loss = tf.abs(column_loss)
        continuity_loss = row_loss + column_loss
        batch_continuity_loss = tf.reduce_mean(continuity_loss)
        tf.summary.scalar('continuity_loss', batch_continuity_loss)
        # Combine the losses to compute the total loss.
        self.loss = batch_ce_loss + self.l1lambda * batch_continuity_loss
        tf.summary.scalar('loss', self.loss)
        self.loss = tf.verify_tensor_all_finite(self.loss, 'Invalid loss detected')

    def set_up_adversarial_loss(self, logits):
        assert logits.get_shape().as_list() == self.labels.get_shape().as_list()
        batch_size, z, y, x, num_channels = logits.get_shape().as_list()
        self.discriminator = DCGAN(input_shape=[y, x, num_channels], dim=2)

        # pred_fake and pred_true are both Tensors of shape [batch_size, 1] conaining
        # the predicted probability that each element in the batch is 'real'.
        pred_fake = self.discriminator.predict_discriminator(logits)
        pred_true = self.discriminator.predict_discriminator(self.labels)

        # We want the network to produce output which fools the discriminator,
        # so we use cross-entropy loss to measure how close the discriminators'
        # predictions are to an array of ONEs (which would indicate it is fooled).

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        generator_loss_batch = cross_entropy(tf.ones_like(pred_fake), pred_fake)
        self.loss = tf.reduce_mean(generator_loss_batch)
        tf.summary.scalar('adversarial_loss', self.loss)
        self.loss = tf.verify_tensor_all_finite(self.loss, 'Invalid loss detected')

        # Compute the discriminator loss
        self.discriminator.discriminator_loss(real_output=pred_true,
                                              fake_output=pred_fake)
        return

    def set_up_loss(self, logit_seed):
        """Set up the loss function of the model."""
        if self.loss_name == "sigmoid_pixelwise":
            self.set_up_sigmoid_pixelwise_loss(logit_seed)
        elif self.loss_name == "l1":
            self.set_up_l1_loss(logit_seed)
        elif self.loss_name == "l1_continuity":
            self.set_up_l1_continuity_loss(logit_seed)
        elif self.loss_name == "ssim":
            self.set_up_ssim_loss(logit_seed)
        elif self.loss_name == "ms_ssim":
            self.set_up_ms_ssim_loss(logit_seed)
        elif self.loss_name == "boundary":
            self.set_up_boundary_loss(logit_seed)
        elif self.loss_name == "adversarial":
            self.set_up_adversarial_loss(logit_seed)
        else:
            raise NotImplementedError

    def set_up_optimizer(self, loss=None, max_gradient_entry_mag=0.7):
        """Sets up the training op for the model."""
        from ffn.training import optimizer
        if loss is None:
            loss = self.loss
        tf.summary.scalar('optimizer_loss', self.loss)

        opt = optimizer.optimizer_from_flags()
        d_opt = tf.train.AdamOptimizer(learning_rate=0.001)

        # grads_and_vars = opt.compute_gradients(loss)
        ffn_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='seed_update')
        d_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=self.discriminator.d_scope_name)
        # Gradients and variables for FFN
        ffn_grads_and_vars = opt.compute_gradients(self.loss, var_list=ffn_trainable_vars)
        d_grads_and_vars = d_opt.compute_gradients(self.discriminator.d_loss,
                                                   var_list=d_trainable_vars)
        for g, v in ffn_grads_and_vars + d_grads_and_vars:
            if g is None:
                tf.logging.error('Gradient is None: %s', v.op.name)

        def _clip_gradients(grads_and_vars):
            if max_gradient_entry_mag > 0.0:
                grads_and_vars = [(tf.clip_by_value(g,
                                                    -max_gradient_entry_mag,
                                                    +max_gradient_entry_mag), v)
                                  for g, v, in grads_and_vars]
            return grads_and_vars

        ffn_grads_and_vars = _clip_gradients(ffn_grads_and_vars)
        d_grads_and_vars = _clip_gradients(d_grads_and_vars)

        trainables = tf.trainable_variables()
        if trainables:
            for var in trainables:
                # tf.summary.histogram(var.name.replace(':0', ''), var)
                tf.summary.histogram(var.name, var)
        for grad, var in ffn_grads_and_vars + d_grads_and_vars:
            # tf.summary.histogram(
            #     'gradients/%s' % var.name.replace(':0', ''), grad)
            tf.summary.histogram(var.name, grad)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.apply_gradients(ffn_grads_and_vars,
                                                global_step=self.global_step,
                                                name='train')
            self.adversarial_train_op = d_opt.apply_gradients(d_grads_and_vars,
                                                              global_step=self.global_step,
                                                              name='train_adversary')

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
            logit_update = _predict_object_mask_2d(
                net, self.depth, self_attention_index=self.self_attention_layer)
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
