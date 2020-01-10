"""Runs FFN inference within a dense bounding box.

Inference is performed within a single process.

Forked from ffn/run_inference.py

usage:
python run_inference.py \
    --bounding_box 'start { x:0 y:0 z:0 } size { x:7601 y:9429 z:1 }' \
    --out_dir results/tmp/ \
    --depth 9 \
    --fov_size 1,${FOV},${FOV} \
    --image "data/test/507727402/507727402_raw.h5:raw" \
    --image_mean 78 \
    --image_stddev 20 \
    --ckpt_id 3052534 \
    --move_threshold 0.07

"""

import logging
import os
import time

from google.protobuf import text_format
from absl import app
from absl import flags
from tensorflow import gfile
import tensorflow as tf

from ffn.utils import bounding_box_pb2
from ffn.inference import inference
# from ffn.inference import inference_flags
from ffn.inference import inference_pb2
from fftracer.utils.config import InferenceConfig

FLAGS = flags.FLAGS

flags.DEFINE_string('bounding_box', None,
                    'BoundingBox proto in text format defining the area '
                    'to be segmented.')
flags.DEFINE_string("out_dir", None, "directory to save to")
flags.DEFINE_integer("depth", 9, "number of residual blocks in model")
flags.DEFINE_list('fov_size', [1, 49, 49], '[z, y, x] size of training fov')
flags.DEFINE_list('deltas', [0, 8, 8], '[z, y, x] size of training fov')
flags.DEFINE_float('image_mean', None,
                   'Mean image intensity to use for input normalization.')
flags.DEFINE_float('image_stddev', None,
                   'Image intensity standard deviation to use for input '
                   'normalization.')
flags.DEFINE_string('image', None, 'The target HDF5 image for segmentation.')
flags.DEFINE_float('move_threshold', None, 'move threshold for ffn inference')
flags.DEFINE_float('lr', 0.001, 'Initial learning rate used to train model.')
flags.DEFINE_string('model_name', 'fftracer.training.models.model.FFNTracerModel',
                    'model name; by default FFN will search for modules relative to '
                    'the FFN module path first, then will search the package list if '
                    'the model class is not found inside ffn.')
flags.DEFINE_integer('min_segment_size', 5, 'minimum segment size; set this low to '
                                            'avoid termination due to small segments.')
flags.DEFINE_float('segment_threshold', 0.075,
                   'threshold to use for auto-generated hard segmentation; set this low'
                   ' to avoid "failed: too small" at end of inference')
flags.DEFINE_list('min_boundary_dist', [0,1,1], 'minimum distance of segments to '
                                                'boundary during inference')
flags.DEFINE_integer('ckpt_id', None,
                 'integer id of the checkpoint to use; by default the script will'
                 ' look in "train_dir/{model_uid}/model.ckpt-{ckpt_id}"')
flags.DEFINE_string('train_dir', 'training-logs',
                    'top-level directory containing subdirectories of model-level '
                    'training logs')
flags.DEFINE_string('seed_policy', 'ManualSeedPolicy',
                    'seed policy to use during inference')

# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(unused_argv):

    move_threshold = FLAGS.move_threshold
    fov_size = dict(zip(["z", "y", "x"], [int(i) for i in FLAGS.fov_size]))
    deltas = dict(zip(["z", "y", "x"], [int(i) for i in FLAGS.deltas]))
    min_boundary_dist = dict(zip(["z", "y", "x"],
                                 [int(i) for i in FLAGS.min_boundary_dist]))
    model_uid = "lr{learning_rate}depth{depth}fov{fov}" \
        .format(learning_rate=FLAGS.lr,
                depth=FLAGS.depth,
                fov=max(fov_size.values())
                )
    segmentation_output_dir = FLAGS.out_dir + model_uid + "mt" + str(move_threshold)
    model_checkpoint_path = "{train_dir}/{model_uid}/model.ckpt-{ckpt_id}"\
        .format(train_dir=FLAGS.train_dir,
                model_uid=model_uid,
                ckpt_id=FLAGS.ckpt_id)

    inference_config = InferenceConfig(
        image=FLAGS.image,
        fov_size=fov_size,
        deltas=deltas,
        depth=FLAGS.depth,
        image_mean=FLAGS.image_mean,
        image_stddev=FLAGS.image_stddev,
        model_checkpoint_path=model_checkpoint_path,
        model_name=FLAGS.model_name,
        segmentation_output_dir=segmentation_output_dir,
        move_threshold=move_threshold,
        min_segment_size=FLAGS.min_segment_size,
        segment_threshold=FLAGS.segment_threshold,
        min_boundary_dist=min_boundary_dist,
        seed_policy=FLAGS.seed_policy
    )
    config = inference_config.to_string()
    logging.info(config)
    req = inference_pb2.InferenceRequest()
    _ = text_format.Parse(config, req)

    if not gfile.Exists(inference_config.segmentation_output_dir):
        gfile.MakeDirs(inference_config.segmentation_output_dir)

    bbox = bounding_box_pb2.BoundingBox()
    text_format.Parse(FLAGS.bounding_box, bbox)

    runner = inference.Runner()
    runner.start(req)

    start_zyx = (bbox.start.z, bbox.start.y, bbox.start.x)
    size_zyx = (bbox.size.z, bbox.size.y, bbox.size.x)
    logging.info("Running; start at {} size {}.".format(start_zyx, size_zyx))

    # Segmentation is attempted from all valid starting points provided by the seed
    # policy by calling runner.canvas.segment_all().
    runner.run(start_zyx,
               size_zyx)
    logging.info("Finished running.")

    counter_path = os.path.join(inference_config.segmentation_output_dir, 'counters.txt')
    if not gfile.Exists(counter_path):
        runner.counters.dump(counter_path)

    runner.stop_executor()
    del runner


if __name__ == '__main__':
    app.run(main)
