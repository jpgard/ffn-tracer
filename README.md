# ffn-tracer
Neuron tracing based on flood-filling networks (FFN)

Example Input Image             |  Ground Truth Trace
:-------------------------:|:-------------------------:
![example neuron](./img/tfrecord_image_raw.jpg)  |  ![example neuron](./img/tfrecord_image_label.jpg)

In order to train a ffn-tracer model, follow these steps:

1. Determine seed locations for each neuron. Seed locations can either be manually determined by inspecting each image using a tool such as [GIMP](https://www.gimp.org/) or Vaa3D, or by a custom algorithm Seed locations should be stored in a CSV file. For an example, see `seed_locations.csv` in this repo.

2. Create `tfrecord`s and training data coordinates for your dataset:

``` 
python generate_mozak_data.py \
    --dataset_ids 507727402 \
    --gs_dir data/gold_standard \
    --img_dir data/img \
    --seed_csv data/seed_locations/seed_locations.csv \
    --out_dir data/tfrecords \
    --num_training_coords 1000
```

This deposits a set of `tfrecord`s containing the training data into `out_dir`, one `tfrecord` per dataset.


3. Train the model.

```python train.py --tfrecord_dir ./data/tfrecords     --out_dir . --coordinate_dir ./data/coords```

4. Run inference on a new dataset.

`[coming soon]`