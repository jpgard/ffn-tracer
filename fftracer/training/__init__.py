"""
Training utilities for ffntracer.
"""

def _get_offset_and_scale_map(image_offset_scale_map=None):
    """From train.py in original ffn repo."""
    if not image_offset_scale_map:
        return {}

    ret = {}
    for vol_def in image_offset_scale_map:
        vol_name, offset, scale = vol_def.split(':')
        ret[vol_name] = float(offset), float(scale)

    return ret


def _get_reflectable_axes(reflectable_axes):
    """From train.py in original ffn repo."""
    return [int(x) + 1 for x in reflectable_axes]


def _get_permutable_axes(permutable_axes):
    """From train.py in original ffn repo."""
    return [int(x) + 1 for x in permutable_axes]
