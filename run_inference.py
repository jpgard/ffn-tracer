"""Runs FFN inference within a dense bounding box.

Inference is performed within a single process.

Forked from ffn/run_inference.py

usage:
python run_inference.py \
    --bounding_box 'start { x:0 y:0 z:0 } size { x:7601 y:9429 z:1 }'
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

# Suppress the annoying tensorflow 1.x deprecation warnings; these make console output
# impossible to parse.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(unused_argv):
    # request = inference_flags.request_from_flags()

    # TODO(jpgard): remove the hard-coded InferenceConfig params, or optionally
    #  move back to command-line parameter as
    #  --inference_request="$(cat configs/inference.pbtxt)" and use
    #  inference.request_from_flags().

    move_threshold = 0.0675
    learning_rate = 0.001
    depth = 9
    fov_size = {"x": 135, "y": 135, "z": 1}
    model_uid = "lr{learning_rate}depth{depth}fov{fov}" \
        .format(learning_rate=learning_rate,
                depth=depth,
                fov=max(v for v in fov_size.values())
                )
    segmentation_output_dir = "results/tmp/" + model_uid + "mt" + str(move_threshold)
    ckpt_id = 3052534
    model_checkpoint_path = "training-logs/{}/model.ckpt-{}".format(model_uid, ckpt_id)

    inference_config = InferenceConfig(
        image="data/test/507727402/507727402_raw.h5:raw",
        fov_size=fov_size,
        deltas={"x": 8, "y": 8, "z": 0}, depth=depth, image_mean=78, image_stddev=20,
        model_checkpoint_path=model_checkpoint_path,
        model_name="fftracer.training.models.model.FFNTracerModel",
        segmentation_output_dir=segmentation_output_dir,
        move_threshold=move_threshold,
        min_segment_size=5,
        segment_threshold=0.075,  # set this low to avoid "failed: too small" at end of
        # inference
        min_boundary_dist={"x": 1, "y": 1, "z": 0},
        seed_policy="ManualSeedPolicy"
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
