"""
Utilities for working with abseil experiment_flags.
"""
import json
from absl import flags


def make_training_flags():
    """Create the experiment_flags for training."""
    # fftracer-specific options
    flags.DEFINE_string("data_dir",
                        "./data", "directory containing the data. It should contain "
                                  "subdirectories for `data_id`, and the `data_id` "
                                  "subdirectory should contain `tfrecords` and `coords` "
                                  "directories.")
    flags.DEFINE_string("data_uid", None, "a unique identifier for this dataset, "
                                         "which also identifies a subdirectory in "
                                         "data_dir containing the tfrecords and coords.")
    flags.DEFINE_boolean("debug", False, "produces debugging output")
    flags.DEFINE_list('permutable_axes', ['1', '2'],
                      'List of integers equal to a subset of [0, 1, 2] specifying '
                      'which of the [z, y, x] axes, respectively, may be permuted '
                      'in order to augment the training data.')

    flags.DEFINE_list('reflectable_axes', ['0', '1', '2'],
                      'List of integers equal to a subset of [0, 1, 2] specifying '
                      'which of the [z, y, x] axes, respectively, may be reflected '
                      'in order to augment the training data.')
    flags.DEFINE_integer('ffn_update_every_iters', None,
                         'update the FFN every `n`th iteration; otherwise update the '
                         'adversary/discriminator.')
    flags.DEFINE_integer('adv_update_every_iters', None,
                         'update the adversary every `n`th iteration; otherwise update '
                         'the FFN.')
    flags.DEFINE_string('adv_args', None,
                        'JSON string with arguments to be passed to the '
                        'adversary/discriminator constructor.')
    # Training infra options (from ffn train.py).
    flags.DEFINE_string('train_base_dir', './training-logs',
                        'Path where checkpoints and other data will be saved into a '
                        'unique '
                        'subdirectory based on experiment hyperparameters.')
    flags.DEFINE_string('master', '', 'Network address of the master.')
    flags.DEFINE_integer('batch_size', 4, 'Number of images in a batch.')
    flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
    flags.DEFINE_integer('ps_tasks', 0, 'Number of tasks in the ps job.')
    flags.DEFINE_integer('max_steps', 10000, 'Number of steps to train for.')
    flags.DEFINE_integer('replica_step_delay', 300,
                         'Require the model to reach step number '
                         '<replica_step_delay> * '
                         '<replica_id> before starting training on a given '
                         'replica.')
    flags.DEFINE_integer('summary_rate_secs', 120,
                         'How often to save summaries (in seconds).')

    # FFN training options (from ffn train.py).
    flags.DEFINE_string('model_args', None,
                        'JSON string with arguments to be passed to the model '
                        'constructor.')
    flags.DEFINE_float('seed_pad', 0.05,
                       'Value to use for the unknown area of the seed.')
    flags.DEFINE_float('threshold', 0.9,
                       'Value to be reached or exceeded at the new center of the '
                       'field of view in order for the network to inspect it.')
    flags.DEFINE_enum('fov_policy', 'fixed', ['fixed', 'max_pred_moves'],
                      'Policy to determine where to move the field of the '
                      'network. "fixed" tries predefined offsets specified by '
                      '"model.shifts". "max_pred_moves" moves to the voxel with '
                      'maximum mask activation within a plane perpendicular to '
                      'one of the 6 Cartesian directions, offset by +/- '
                      'model.deltas from the current FOV position.')
    flags.DEFINE_integer('fov_moves', 1,
                         'Number of FOV moves by "model.delta" voxels to execute '
                         'in every dimension. Currently only works with the '
                         '"max_pred_moves" policy.')
    flags.DEFINE_boolean('shuffle_moves', True,
                         'Whether to randomize the order of the moves used by the '
                         'network with the "fixed" policy.')
    flags.DEFINE_list('image_offset_scale_map', None,
                      'Optional per-volume specification of mean and stddev. '
                      'Every entry in the list is a colon-separated tuple of: '
                      'volume_label, offset, scale.')
    flags.DEFINE_float('image_mean', None,
                       'Mean image intensity to use for input normalization.')
    flags.DEFINE_float('image_stddev', None,
                       'Image intensity standard deviation to use for input '
                       'normalization.')


def uid_from_flags(flags):
    """Construct a uique string identifier for an experiment."""
    model_args = json.loads(flags.model_args)
    uid = "bs{batch_size}lr{learning_rate}opt{optimizer}data{data_uid}".format(
        batch_size=flags.batch_size,
        learning_rate=flags.learning_rate,
        optimizer=flags.optimizer,
        data_uid=flags.data_uid
    )
    # model_args are handled separately; these require some extra logic to parse
    uid += "fov{}".format("".join([str(x) for x in model_args.pop("fov_size")]))
    uid += "loss{}".format(model_args.pop("loss_name"))
    uid += "".join([str(k) + str(v) for k,v in model_args.items()])
    if flags.adv_args:
        adv_args = json.loads(flags.adv_args)
        uid += "_adv_"
        uid += "".join([str(k) + str(v) for k, v in adv_args.items()])
        assert not (flags.ffn_update_every_iters and flags.adv_update_every_iters)
        if flags.ffn_update_every_iters:
            uid += "ffnupdate{}".format(flags.ffn_update_every_iters)
        elif flags.adv_update_every_iters:
            uid += "advupdate{}".format(flags.adv_update_every_iters)
    return uid
