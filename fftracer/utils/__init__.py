import tensorflow as tf

# allowed image extensions
VALID_IMAGE_EXTENSIONS = (".jpg", ".png")

# a dictionary describing the features from mozak data
FEATURE_SCHEMA = {
    'shape_x': tf.io.FixedLenFeature(1, tf.int64),
    'shape_y': tf.io.FixedLenFeature(1, tf.int64),
    'seed_x': tf.io.FixedLenFeature(1, tf.int64),
    'seed_y': tf.io.FixedLenFeature(1, tf.int64),
    'seed_z': tf.io.FixedLenFeature(1, tf.int64),
    'image_raw': tf.io.VarLenFeature(tf.int64),
    'image_label': tf.io.VarLenFeature(tf.float32),
}
