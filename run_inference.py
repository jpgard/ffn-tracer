"""Runs FFN inference within a dense bounding box.

Inference is performed within a single process.

Forked from ffn/run_inference.py

usage:
python run_inference.py \
    --bounding_box 'start { x:0 y:0 z:0 } size { x:7601 y:9429 z:1 }' \
    --out_dir results/507727402/ \
    --depth 9 \
    --fov_size 1,${FOV},${FOV} \
    --image "data/test/507727402/507727402_raw.h5:raw" \
    --image_mean 78 \
    --image_stddev 20 \
    --ckpt_id 10000000 \
    --move_threshold 0.12 \
    --seed_policy TipTracerSeedPolicy

"""

import logging
import os
import shutil
import tempfile
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
flags.DEFINE_string("out_dir", None, "relative path of directory to save to")
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
logging.basicConfig(level=logging.DEBUG)

def copy_file_to_tempdir(fp, dirname):
    """Copy the file at out_dir to dirname, keeping the same base name.
    Returns the path to the new file, as string.
    """
    fp, dset_name = fp.split(":")
    assert os.path.isdir(dirname)
    assert os.path.exists(fp)
    fsize = os.path.getsize(fp)
    if fsize > 1e8:
        logging.warning("Copying large file; size in bytes: {}".format(fsize))
    filename = os.path.basename(fp)
    shutil.copy(fp, dirname)
    dest_fp = os.path.join(dirname, filename)
    return dest_fp + ":" + dset_name


def main(unused_argv):
    move_threshold = FLAGS.move_threshold
    fov_size = dict(zip(["z", "y", "x"], [int(i) for i in FLAGS.fov_size]))
    deltas = dict(zip(["z", "y", "x"], [int(i) for i in FLAGS.deltas]))
    min_boundary_dist = dict(zip(["z", "y", "x"],
                                 [int(i) for i in FLAGS.min_boundary_dist]))
    model_uid = "lr{learning_rate}depth{depth}fov{fov}" \
        .format(learning_rate=FLAGS.lr,
                depth=FLAGS.depth,
                fov=max(fov_size.values()),
                )
    segmentation_output_dir = os.path.join(
        os.getcwd(),
        FLAGS.out_dir + model_uid + "mt" + str(move_threshold) + "policy" +
        FLAGS.seed_policy
    )
    model_checkpoint_path = "{train_dir}/{model_uid}/model.ckpt-{ckpt_id}"\
        .format(train_dir=FLAGS.train_dir,
                model_uid=model_uid,
                ckpt_id=FLAGS.ckpt_id)
    if not gfile.Exists(segmentation_output_dir):
        gfile.MakeDirs(segmentation_output_dir)
    else:
        logging.warning(
            "segmentation_output_dir {} already exists; this may cause inference to "
            "terminate without running.".format(segmentation_output_dir))

    with tempfile.TemporaryDirectory(dir=segmentation_output_dir) as tmpdir:

        # Create a temporary local copy of the HDF5 image, because simulataneous access
        # to HDF5 files is not allowed (only recommended for small files).

        temp_image = copy_file_to_tempdir(FLAGS.image, tmpdir)

        inference_config = InferenceConfig(
            image=temp_image,
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
                   size_zyx,
                   allow_overlapping_segmentation=True,
                   # reset_seed_per_segment=False,
                   # keep_history=True  # this only keeps seed history; not that useful
                   )
        logging.info("Finished running.")

        counter_path = os.path.join(inference_config.segmentation_output_dir, 'counters.txt')
        if not gfile.Exists(counter_path):
            runner.counters.dump(counter_path)

        runner.stop_executor()
        del runner


if __name__ == '__main__':
    app.run(main)
