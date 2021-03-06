"""
Train a flood-filling tracer model.

This script mostly follows the logic of the original ffn train.py script, with custom
data loading for mozak/allen institute imaging.

usage:
export OPTIMIZER="adam";LOSS="ot";FOV=49
export GPU_ID="3"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID
export DATA="clean-02-2020"
export DATA="blurry"

python train.py \
    --data_uid $DATA
    --image_mean 78 --image_stddev 20 \
    --max_steps 10000000 \
    --optimizer $OPTIMIZER \
    --model_args "{\"fov_size\": [${FOV}, ${FOV}, 1], \"loss_name\": \"$LOSS\", \"ot_niters\": 100}" \

"""

from collections import deque
from functools import partial
import itertools
import json
import logging
import os
import random
import time

from absl import flags
from absl import app
import numpy as np
import six
from scipy.special import expit, logit
import tensorflow as tf
from ffn.training import augmentation, mask
from ffn.training import optimizer  # Necessary so that optimizer flags are defined.

from fftracer.training import input
from fftracer.training import _get_offset_and_scale_map, _get_permutable_axes, \
    _get_reflectable_axes
from fftracer.training.models.model import FFNTracerModel
from fftracer.training.input import offset_and_scale_patches
from fftracer.training.evaluation import EvalTracker
from fftracer.utils.debugging import write_patch_and_label_to_img
from fftracer.utils.flags import uid_from_flags, make_training_flags

FLAGS = flags.FLAGS
make_training_flags()

# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def is_adversary_update_step(step, model) -> bool:
    is_adversarial_model = hasattr(model, "adversarial_train_op") and \
                           getattr(model, "adversarial_train_op") is not None
    if not is_adversarial_model:
        return False
    assert not (FLAGS.ffn_update_every_iters and FLAGS.adv_update_every_iters), \
        "Only specify one of --ffn_update_every_iters and --adv_update_every_iters."
    if FLAGS.ffn_update_every_iters:  # check whether this is an update step
        return step % FLAGS.ffn_update_every_iters != 0
    if FLAGS.adv_update_every_iters:
        return step % FLAGS.adv_update_every_iters == 0
    else:
        raise NotImplementedError(
            "If using an adversarial model you must specify one of  "
            "--ffn_update_every_iters or --adv_update_every_iters")

def _get_offset_and_scale_map():
    if not FLAGS.image_offset_scale_map:
        return {}
    ret = {}
    for vol_def in FLAGS.image_offset_scale_map:
        vol_name, offset, scale = vol_def.split(':')
        ret[vol_name] = float(offset), float(scale)

    return ret


def fov_moves():
    """From ffn train.py ."""
    # Add one more move to get a better fill of the evaluation area.
    if FLAGS.fov_policy == 'max_pred_moves':
        return FLAGS.fov_moves + 1
    return FLAGS.fov_moves


def train_labels_size(model):
    """From ffn train.py ."""
    return (np.array(model.pred_mask_size) +
            np.array(model.deltas) * 2 * fov_moves())


def train_eval_size(model):
    """From ffn train.py ."""
    return (np.array(model.pred_mask_size) +
            np.array(model.deltas) * 2 * FLAGS.fov_moves)


def train_image_size(model):
    """From ffn train.py ."""
    return (np.array(model.input_image_size) +
            np.array(model.deltas) * 2 * fov_moves())


def train_canvas_size(model):
    """From ffn train.py ."""
    return (np.array(model.input_seed_size) +
            np.array(model.deltas) * 2 * fov_moves())


def fixed_offsets(model, seed, fov_shifts=None):
    """Generates offsets based on a fixed list.

    From ffn train.py .
    """
    for off in itertools.chain([(0, 0, 0)], fov_shifts):
        if model.dim == 3:
            is_valid_move = seed[:,
                            seed.shape[1] // 2 + off[2],
                            seed.shape[2] // 2 + off[1],
                            seed.shape[3] // 2 + off[0],
                            0] >= logit(FLAGS.threshold)
        else:
            is_valid_move = seed[:,
                            seed.shape[1] // 2 + off[1],
                            seed.shape[2] // 2 + off[0],
                            0] >= logit(FLAGS.threshold)

        if not is_valid_move:
            continue

        yield off


def max_pred_offsets(model, seed):
    """Generates offsets with the policy used for inference."""
    # Always start at the center.
    queue = deque([(0, 0, 0)])
    done = set()

    train_image_radius = train_image_size(model) // 2
    input_image_radius = np.array(model.input_image_size) // 2

    while queue:
        offset = queue.popleft()

        # Drop any offsets that would take us beyond the image fragment we
        # loaded for training.
        if np.any(np.abs(np.array(offset)) + input_image_radius >
                  train_image_radius):
            continue

        # Ignore locations that were visited previously.
        quantized_offset = (
            offset[0] // max(model.deltas[0], 1),
            offset[1] // max(model.deltas[1], 1),
            offset[2] // max(model.deltas[2], 1))

        if quantized_offset in done:
            continue

        done.add(quantized_offset)

        yield offset

        # Look for new offsets within the updated seed.
        curr_seed = mask.crop_and_pad(seed, offset, model.pred_mask_size[::-1])
        todos = sorted(
            movement.get_scored_move_offsets(
                model.deltas[::-1],
                curr_seed[0, ..., 0],
                threshold=logit(FLAGS.threshold)), reverse=True)
        queue.extend((x[2] + offset[0],
                      x[1] + offset[1],
                      x[0] + offset[2]) for _, x in todos)


def get_example(load_example, eval_tracker, model, get_offsets):
    """Generates individual training examples.

    Args:
      load_example: callable returning a tuple of image and label ndarrays
                    as well as the seed coordinate and volume name of the example
      eval_tracker: EvalTracker object
      model: FFNModel object
      get_offsets: iterable of (x, y, z) offsets to investigate within the
          training patch

    Yields:
      tuple of:
        seed array, shape [1, z, y, x, 1]
        image array, shape [1, z, y, x, 1]
        label array, shape [1, z, y, x, 1]
    """
    seed_shape = train_canvas_size(model).tolist()[::-1]
    while True:
        full_patches, full_labels, loss_weights, coord, volname = load_example()
        # Write a random fraction of paired examples to images and make sure they have
        # matching and correct orientations.
        if FLAGS.debug:
            if random.uniform(0, 1) > 0.999:
                write_patch_and_label_to_img(
                    patch=full_patches[0, 0, :, :, 0] * FLAGS.image_stddev + FLAGS.image_mean,
                    label=full_labels[0, 0, :, :, 0] * 255,
                    unique_id='_'.join(coord[0].astype(str).tolist()),
                    dirname="./debug"
                )

        # Always start with a clean seed.
        seed = logit(mask.make_seed(seed_shape, 1, pad=FLAGS.seed_pad))
        for off in get_offsets(model, seed):
            predicted = mask.crop_and_pad(seed, off, model.input_seed_size[::-1])
            patches = mask.crop_and_pad(full_patches, off, model.input_image_size[::-1])
            labels = mask.crop_and_pad(full_labels, off, model.pred_mask_size[::-1])
            weights = mask.crop_and_pad(loss_weights, off, model.pred_mask_size[::-1])

            # Necessary, since the caller is going to update the array and these
            # changes need to be visible in the following iterations.
            assert predicted.base is seed
            yield predicted, patches, labels, weights
        # TODO(jpgard): track volname in eval_tracker. Currently nothing is done with
        #  volname, but it should be monitored to ensure coverage of all training
        #  volumes. Similar for coord; would be good to check coordinates covered,
        #  or at least a sample of them.
        eval_tracker.add_patch(
            full_labels, seed, loss_weights, coord, volname, full_patches)


def get_batch(load_example, eval_tracker, model, batch_size, get_offsets):
    """Generates batches of training examples.

    Args:
      load_example: callable returning a tuple of image and label ndarrays
                    as well as the seed coordinate and volume name of the example
      eval_tracker: EvalTracker object
      model: FFNModel object
      batch_size: desidred batch size
      get_offsets: iterable of (x, y, z) offsets to investigate within the
          training patch

    Yields:
      tuple of:
        seed array, shape [b, z, y, x, 1]
        image array, shape [b, z, y, x, 1]
        label array, shape [b, z, y, x, 1]

      where 'b' is the batch_size.
    """
    def _batch(iterable):
        for batch_vals in iterable:
            # `batch_vals` is sequence of `batch_size` tuples returned by the
            # `get_example` generator, to which we apply the following transformation:
            #   [(a0, b0), (a1, b1), .. (an, bn)] -> [(a0, a1, .., an),
            #                                         (b0, b1, .., bn)]
            # (where n is the batch size) to get a sequence, each element of which
            # represents a batch of values of a given type (e.g., seed, image, etc.)
            yield zip(*batch_vals)

    # Create a separate generator for every element in the batch. This generator
    # will automatically advance to a different training example once the allowed
    # moves for the current location are exhausted.
    for seeds, patches, labels, weights in _batch(six.moves.zip(
            *[get_example(load_example, eval_tracker, model, get_offsets) for _
              in range(batch_size)])):

        batched_seeds = np.concatenate(seeds)
        yield (batched_seeds, np.concatenate(patches), np.concatenate(labels),
               np.concatenate(weights))

        # batched_seed is updated in place with new predictions by the code
        # calling get_batch. Here we distribute these updated predictions back
        # to the buffer of every generator.
        for i in range(batch_size):
            seeds[i][:] = batched_seeds[i, ...]


def run_training_step(sess, model, fetch_summary, feed_dict):
    """Runs one training step for a single FFN FOV."""
    ops_to_run = [model.train_op, model.global_step, model.logits]

    if fetch_summary is not None:
        ops_to_run.append(fetch_summary)
    results = sess.run(ops_to_run, feed_dict)
    step, prediction = results[1:3]
    if fetch_summary is not None:
        summ = results[-1]
    else:
        summ = None

    return prediction, step, summ


def run_adversary_training_step(sess, model, fetch_summary, feed_dict):
    ops_to_run = [model.adversarial_train_op, model.global_step, model.logits]

    if fetch_summary is not None:
        ops_to_run.append(fetch_summary)
    results = sess.run(ops_to_run, feed_dict)
    step = results[1]
    if fetch_summary is not None:
        summ = results[-1]
    else:
        summ = None

    return step, summ


def define_data_input(model, queue_batch=None):
    """Adds TF ops to load input data.
    Mimics structure of function of the same name in ffn.train.py
    """
    permutable_axes = np.array(FLAGS.permutable_axes, dtype=np.int32)
    reflectable_axes = np.array(FLAGS.reflectable_axes, dtype=np.int32)

    tfrecord_dir = os.path.join(FLAGS.data_dir, FLAGS.data_uid, "tfrecords")
    image_volume_map, label_volume_map = input.load_img_and_label_maps_from_tfrecords(
        tfrecord_dir)
    # write the datasets to debugging directory
    if FLAGS.debug:
        for dataset_id in image_volume_map.keys():
            write_patch_and_label_to_img(
                image_volume_map[dataset_id].astype(np.uint8),
                (label_volume_map[dataset_id] * 256).astype(np.uint8),
                str(dataset_id),
                "./debug"
            )

    # Fetch (x,y,z) sizes of images and labels; coerce to list to avoid unintentional
    # numpy broadcasting when intended behavior is concatenation. The size of
    # images/labels are the FOV with an added margin of (deltas * fov_moves) along each
    # axis.
    label_size_xyz = train_labels_size(model).tolist()
    image_size_xyz = train_image_size(model).tolist()
    # Fetch a single coordinate and volume name from a queue reading the
    # coordinate files or from saved hard/important examples
    coordinate_dir = os.path.join(FLAGS.data_dir, FLAGS.data_uid, "coords")
    coord_zyx, volname = input.load_patch_coordinates(coordinate_dir)
    # Coordinates are stored in zyx order but are used in xyz order for loading functions,
    # so they need to be reversed.
    coord_xyz = tf.reverse(coord_zyx, [False, True])

    # Note: label_size_xyz and image_size_xyz are reversed in call to
    # load_from_numpylike_2d() to match orientation of axes in coordinate and sizes.

    patch = input.load_from_numpylike_2d(coord_xyz, volname, shape=image_size_xyz,
                                         volume_map=image_volume_map, name="LoadPatch")
    labels = input.load_from_numpylike_2d(coord_xyz, volname, shape=label_size_xyz,
                                          volume_map=label_volume_map, name="LoadLabels")
    # Give labels shape [batch_size, z, y, x, n_channels]
    label_shape = [1] + label_size_xyz[::-1] + [1]

    with tf.name_scope(None, 'ReshapeLabels', [labels, label_shape]) as scope:
        labels = tf.reshape(labels, label_shape)

    loss_weights = tf.constant(np.ones(label_shape, dtype=np.float32))

    # Give images shape [batch_size, z, y, x, n_channels]
    data_shape = [1] + image_size_xyz[::-1] + [1]

    with tf.name_scope(None, 'ReshapePatch', [patch, data_shape]) as scope:
        patch = tf.reshape(patch, data_shape)

    if ((FLAGS.image_stddev is None or FLAGS.image_mean is None) and
            not FLAGS.image_offset_scale_map):
        raise ValueError('--image_mean, --image_stddev or --image_offset_scale_map '
                         'need to be defined')

    # Apply basic augmentations.
    transform_axes = augmentation.PermuteAndReflect(
        rank=5, permutable_axes=_get_permutable_axes(permutable_axes),
        reflectable_axes=_get_reflectable_axes(reflectable_axes))
    labels = transform_axes(labels)
    patch = transform_axes(patch)
    loss_weights = transform_axes(loss_weights)
    # Normalize image data.
    patch = offset_and_scale_patches(
        patch, volname,
        offset_scale_map=_get_offset_and_scale_map(),
        default_offset=FLAGS.image_mean,
        default_scale=FLAGS.image_stddev)

    ## Create a batch of examples. Note that any TF operation before this line
    ## will be hidden behind a queue, so expensive/slow ops can take advantage
    ## of multithreading.

    patches, labels, loss_weights = tf.train.shuffle_batch(
        [patch, labels, loss_weights],
        queue_batch,
        capacity=32 * FLAGS.batch_size,
        min_after_dequeue=4 * FLAGS.batch_size,
        num_threads=max(1, FLAGS.batch_size // 2),
        enqueue_many=True
    )
    return patches, labels, loss_weights, coord_zyx, volname


def prepare_ffn(model):
    """Creates the TF graph for an FFN.

    Ported from ffn.train.py.
    """
    shape = [FLAGS.batch_size] + list(model.pred_mask_size[::-1]) + [1]
    model.labels = tf.placeholder(tf.float32, shape, name='labels')
    model.loss_weights = tf.placeholder(tf.float32, shape, name='loss_weights')
    model.define_tf_graph()


def main(argv):
    experiment_uid = uid_from_flags(FLAGS)
    train_dir = os.path.join(FLAGS.train_base_dir, experiment_uid)
    with tf.Graph().as_default():
        with tf.device(
                tf.train.replica_device_setter(FLAGS.ps_tasks, merge_devices=True)):
            # The constructor might define TF ops/placeholders, so it is important
            # that the FFN is instantiated within the current context.

            # Note: all inputs to ffn.training.model.FFNModel are in format (x, y, z),
            # except the fov_size, for historical reasons.

            # If fov_size is specified at command line, it will be stored as a list of
            # strings; these need to be coerced to integers.

            model = FFNTracerModel(batch_size=FLAGS.batch_size,
                                   adv_args=FLAGS.adv_args,
                                   **json.loads(FLAGS.model_args))
            eval_shape_zyx = train_eval_size(model).tolist()[::-1]

            eval_tracker = EvalTracker(eval_shape_zyx)
            load_data_ops = define_data_input(model, queue_batch=1)
            prepare_ffn(model)
            merge_summaries_op = tf.summary.merge_all()

            # if FLAGS.task == 0:
            #     save_flags()

            summary_writer = None
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.25)
            scaffold = tf.train.Scaffold(saver=saver)
            with tf.train.MonitoredTrainingSession(
                    master=FLAGS.master,
                    is_chief=(FLAGS.task == 0),
                    save_summaries_steps=None,
                    save_checkpoint_secs=300,
                    config=tf.ConfigProto(
                        log_device_placement=False,
                        allow_soft_placement=True,
                        gpu_options=tf.GPUOptions(
                            allow_growth=True,
                            # visible_device_list=FLAGS.visible_gpus
                        )
                    ),
                    checkpoint_dir=train_dir,
                    scaffold=scaffold) as sess:

                eval_tracker.sess = sess
                step = int(sess.run(model.global_step))

                if FLAGS.task > 0:
                    # To avoid early instabilities when using multiple replicas, we use
                    # a launch schedule where new replicas are brought online gradually.
                    logging.info('Delaying replica start.')
                    while step < FLAGS.replica_step_delay * FLAGS.task:
                        time.sleep(5.0)
                        step = int(sess.run(model.global_step))
                else:
                    summary_writer = tf.summary.FileWriterCache.get(train_dir)
                    summary_writer.add_session_log(
                        tf.summary.SessionLog(status=tf.summary.SessionLog.START), step)

                fov_shifts_xyz = list(model.shifts)
                if FLAGS.shuffle_moves:
                    random.shuffle(fov_shifts_xyz)

                # Policy_map is a dict mapping a fov_policy to a callable that
                # generates offsets, given a model and a seed as inputs. Note that
                # 'fixed' is the policy used for training, and max_pred_moves is the
                # policy used for inference.

                policy_map = {
                    'fixed': partial(fixed_offsets, fov_shifts=fov_shifts_xyz),
                    'max_pred_moves': max_pred_offsets
                }

                # JG: batch_it contains (seed, image, label, weights), where each is of
                # shape [b, z, y, x, 1]
                batch_it = get_batch(lambda: sess.run(load_data_ops),
                                     eval_tracker, model, FLAGS.batch_size,
                                     policy_map[FLAGS.fov_policy])

                t_last = time.time()

                while not sess.should_stop() and step < FLAGS.max_steps:
                    # Run summaries periodically.
                    t_curr = time.time()
                    if t_curr - t_last > FLAGS.summary_rate_secs and FLAGS.task == 0:
                        summ_op = merge_summaries_op
                        t_last = t_curr
                    else:
                        summ_op = None

                    seed, patches, labels, weights = next(batch_it)
                    # JG: weights, labels, patches, and seed all have
                    # shape [b, z, y, x, 1] at this stage.

                    feed_dict = {
                                model.loss_weights: weights,
                                model.labels: labels,
                                model.input_patches: patches,
                                model.input_seed: seed,
                            }

                    if is_adversary_update_step(step, model):
                        # Update the adversary
                        step, summ = run_adversary_training_step(
                            sess, model, summ_op,
                            feed_dict=feed_dict
                        )
                    else:
                        # Update the FFN model
                        updated_seed, step, summ = run_training_step(
                            sess, model, summ_op,
                            feed_dict=feed_dict)

                        # Save prediction results in the original seed array so that
                        # they can be used in subsequent steps.
                        mask.update_at(seed, (0, 0, 0), updated_seed)

                    # Record summaries.
                    if summ is not None:
                        logging.info('Saving summaries.')
                        summ = tf.Summary.FromString(summ)

                        # Compute a loss over the whole training patch (i.e. more than a
                        # single-step field of view of the network). This quantifies the
                        # quality of the final object mask.
                        summ.value.extend(eval_tracker.get_summaries())
                        eval_tracker.reset()

                        assert summary_writer is not None
                        summary_writer.add_summary(summ, step)

            if summary_writer is not None:
                summary_writer.flush()


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    app.run(main)
