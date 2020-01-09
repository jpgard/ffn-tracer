"""Functions for creating configurations."""

class InferenceConfig:
    def __init__(self, image, fov_size: dict, deltas: dict, depth: int, image_mean: int,
                 image_stddev: int, model_checkpoint_path: str, model_name: str,
                 segmentation_output_dir: str, init_activation: float = 0.95,
                 pad_value: float = 0.05, move_threshold: float = 0.9,
                 min_boundary_dist: dict = {"x": 1, "y": 1, "z": 1},
                 segment_threshold: float = 0.6, min_segment_size: int = 1000,
                 checkpoint_interval: int = 1800,
                 seed_policy: str = "PolicyPeaks"):
        for d in (fov_size, deltas, min_boundary_dist):
            assert all([i in d.keys() for i in ("x", "y", "z")]), \
                "dict is missing one or more expected keys: {}".format(d)
        self.image = image
        self.fov_size = fov_size
        self.deltas = deltas
        self.depth = depth
        self.image_mean = image_mean
        self.image_stddev = image_stddev
        self.model_checkpoint_path = model_checkpoint_path
        self.model_name = model_name
        self.segmentation_output_dir = segmentation_output_dir
        self.init_activation = init_activation
        self.pad_value = pad_value
        self.move_threshold = move_threshold
        self.min_boundary_dist = min_boundary_dist
        self.segment_threshold = segment_threshold
        self.min_segment_size = min_segment_size
        self.checkpoint_interval = checkpoint_interval
        self.seed_policy = seed_policy

    def to_string(self):
        """Create a config string which can be parsed with text_format.Parse()."""
        config = '''image {{
          hdf5: "{image}"
        }}
        image_mean: {image_mean}
        image_stddev: {image_stddev}
        checkpoint_interval: {checkpoint_interval}
        seed_policy: "{seed_policy}"
        model_checkpoint_path: "{model_checkpoint_path}"
        model_name: "{model_name}"
        model_args: "{{\\"depth\\": {depth}, \\"fov_size\\": [{fov_size[x]}, {fov_size[y]}, {fov_size[z]}], 
        \\"deltas\\": [{deltas[x]}, {deltas[y]}, {deltas[z]}] }}"
        segmentation_output_dir: "{segmentation_output_dir}"
        inference_options {{
          init_activation: {init_activation}
          pad_value: {pad_value}
          move_threshold: {move_threshold}
          min_boundary_dist {{ x: {min_boundary_dist[x]} y: {min_boundary_dist[y]} z: {min_boundary_dist[z]} }}
          segment_threshold: {segment_threshold}
          min_segment_size: {min_segment_size}
        }}'''.format(
            image=self.image, image_mean=self.image_mean,
            image_stddev=self.image_stddev, checkpoint_interval=self.checkpoint_interval,
            seed_policy=self.seed_policy,
            model_checkpoint_path=self.model_checkpoint_path, model_name=self.model_name,
            depth=self.depth, fov_size=self.fov_size, deltas=self.deltas,
            segmentation_output_dir=self.segmentation_output_dir,
            init_activation=self.init_activation, pad_value=self.pad_value,
            move_threshold=self.move_threshold, min_boundary_dist=self.min_boundary_dist,
            segment_threshold=self.segment_threshold,
            min_segment_size=self.min_segment_size
        )
        return config
