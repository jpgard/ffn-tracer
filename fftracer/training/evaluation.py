"""
Utilities for evaluation.

Ported from ffn train.py .
"""
import tensorflow as tf
from scipy.special import logit, expit
from collections import deque
from io import BytesIO

import PIL
import PIL.Image
import numpy as np

from ffn.training import mask


class EvalTracker(object):
    """Tracks eval results over multiple training steps."""

    def __init__(self, eval_shape):
        self.eval_labels = tf.placeholder(
            tf.float32, [1] + eval_shape + [1], name='eval_labels')
        self.eval_preds = tf.placeholder(
            tf.float32, [1] + eval_shape + [1], name='eval_preds')
        self.eval_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.eval_preds, labels=self.eval_labels))
        self.reset()
        self.eval_threshold = logit(0.9)
        self.sess = None
        self._eval_shape = eval_shape

    def reset(self):
        """Resets status of the tracker."""
        self.loss = 0
        self.num_patches = 0
        self.tp = 0
        self.tn = 0
        self.fn = 0
        self.fp = 0
        self.total_voxels = 0
        self.masked_voxels = 0
        self.images_xy = deque(maxlen=16)
        self.images_xz = deque(maxlen=16)
        self.images_yz = deque(maxlen=16)

    def slice_image(self, labels, predicted, weights, inputs, slice_axis):
        """Builds a tf.Summary showing a slice of an object mask.

        The object mask slice is shown side by side with the corresponding
        ground truth mask.

        Args:
          labels: ndarray of ground truth data, shape [1, z, y, x, 1]
          predicted: ndarray of predicted data, shape [1, z, y, x, 1]
          weights: ndarray of loss weights, shape [1, z, y, x, 1]
          inputs: ndarray of input images, shape [1, z, y, x, 1]
          slice_axis: axis in the middle of which to place the cutting plane
              for which the summary image will be generated, valid values are
              2 ('x'), 1 ('y'), and 0 ('z').

        Returns:
          tf.Summary.Value object with the image.
        """
        zyx = list(labels.shape[1:-1])
        selector = [0, slice(None), slice(None), slice(None), 0]
        selector[slice_axis + 1] = zyx[slice_axis] // 2
        selector = tuple(selector)

        del zyx[slice_axis]
        h, w = zyx

        buf = BytesIO()
        labels = (labels[selector] * 255).astype(np.uint8)
        predicted = (predicted[selector] * 255).astype(np.uint8)
        weights = (weights[selector] * 255).astype(np.uint8)
        inputs = (inputs[selector] * 255).astype(np.uint8)

        im = PIL.Image.fromarray(np.concatenate([labels, predicted,
                                                 weights, inputs], axis=1), 'L')
        im.save(buf, 'PNG')

        axis_names = 'zyx'
        axis_names = axis_names.replace(axis_names[slice_axis], '')

        return tf.Summary.Value(
            tag='final_%s' % axis_names[::-1],
            image=tf.Summary.Image(
                height=h, width=w * 3, colorspace=1,  # greyscale
                encoded_image_string=buf.getvalue()))

    def add_patch(self, labels, predicted, weights,
                  coord=None, volname=None, patches=None):
        """Evaluates single-object segmentation quality."""
        predicted = mask.crop_and_pad(predicted, (0, 0, 0), self._eval_shape)
        weights = mask.crop_and_pad(weights, (0, 0, 0), self._eval_shape)
        labels = mask.crop_and_pad(labels, (0, 0, 0), self._eval_shape)
        inputs = mask.crop_and_pad(patches, (0, 0, 0), self._eval_shape)
        loss, = self.sess.run([self.eval_loss], {self.eval_labels: labels,
                                                 self.eval_preds: predicted})
        self.loss += loss
        self.total_voxels += labels.size
        self.masked_voxels += np.sum(weights == 0.0)

        pred_mask = predicted >= self.eval_threshold
        true_mask = labels > 0.5
        pred_bg = np.logical_not(pred_mask)
        true_bg = np.logical_not(true_mask)

        self.tp += np.sum(pred_mask & true_mask)
        self.fp += np.sum(pred_mask & true_bg)
        self.fn += np.sum(pred_bg & true_mask)
        self.tn += np.sum(pred_bg & true_bg)
        self.num_patches += 1

        predicted = expit(predicted)
        self.images_xy.append(self.slice_image(labels, predicted, weights, inputs, 0))
        self.images_xz.append(self.slice_image(labels, predicted, weights, inputs, 1))
        self.images_yz.append(self.slice_image(labels, predicted, weights, inputs, 2))

    def get_summaries(self):
        """Gathers tensorflow summaries into single list."""

        if not self.total_voxels:
            return []

        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)

        for images in self.images_xy, self.images_xz, self.images_yz:
            for i, summary in enumerate(images):
                summary.tag += '/%d' % i

        summaries = (
                list(self.images_xy) + list(self.images_xz) + list(self.images_yz) + [
            tf.Summary.Value(tag='masked_voxel_fraction',
                             simple_value=(self.masked_voxels /
                                           self.total_voxels)),
            tf.Summary.Value(tag='eval/patch_loss',
                             simple_value=self.loss / self.num_patches),
            tf.Summary.Value(tag='eval/patches',
                             simple_value=self.num_patches),
            tf.Summary.Value(tag='eval/accuracy',
                             simple_value=(self.tp + self.tn) / (
                                     self.tp + self.tn + self.fp + self.fn)),
            tf.Summary.Value(tag='eval/precision',
                             simple_value=precision),
            tf.Summary.Value(tag='eval/recall',
                             simple_value=recall),
            tf.Summary.Value(tag='eval/specificity',
                             simple_value=self.tn / max(self.tn + self.fp, 1)),
            tf.Summary.Value(tag='eval/f1',
                             simple_value=(2.0 * precision * recall /
                                           (precision + recall)))
        ])

        return summaries
