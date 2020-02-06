"""
Utilities for working with abseil flags.
"""
import json

def uid_from_flags(flags):
    """Construct a uique string identifier for an experiment."""
    model_args = json.loads(flags.model_args)
    uid = "bs{batch_size}lr{learning_rate}opt{optimizer}".format(
        batch_size=flags.batch_size,
        learning_rate=flags.learning_rate,
        optimizer=flags.optimizer
    )
    # model_args are handled separately; these require some extra logic to parse
    uid += "fov{}".format("".join([str(x) for x in model_args.pop("fov_size")]))
    uid += "loss{}".format(model_args.pop("loss_name"))
    uid += "deltas{}".format("".join([str(x) for x in model_args.pop("deltas")]))
    uid += "".join([str(k) + str(v) for k,v in model_args.items()])
    if flags.adv_args:
        adv_args = json.loads(flags.adv_args)
        uid += "_adv_"
        uid += "".join([str(k) + str(v) for k, v in adv_args.items()])
    return uid
