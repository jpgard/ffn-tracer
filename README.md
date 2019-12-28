# ffn-tracer
Neuron tracing based on flood-filling networks (FFN)


![example neuron](./img/patch_and_label_507727402_f32.png)

Before executing the steps below, set up and activate a virtual environment using the `requirements.txt` file.

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
    
    This deposits a set of `tfrecord`s containing the training data
    into `out_dir`, one `tfrecord` per dataset.
    
3. *Training*:

    a. Initiate model training. You should determine values for `image_mean` and `image_stddev` for your data.
    
    ```
    python train.py --tfrecord_dir ./data/tfrecords \
        --out_dir . --coordinate_dir ./data/coords \
         --image_mean 78 --image_stddev 20 --train_dir ./training-logs \
         --max_steps 1000000
    ```
    
    b. (**optional, but recommended**) initiate TensorBoard to monitor training and view sample labeled images:
    
    `tensorboard --logdir ./training-logs`
    
    Note that if you are running training on a remote server, in order to view the TensorBoard output in your browser, you will first need to run `ssh -N -f -L localhost:16006:localhost:6006 user@hostname`.

4. *Inference*: Run inference on a new dataset.

  a. Generate the target volume as hdf5. Note that the working directory should contain a set of z-slice images as `.png` files. These are assembled into the test volume by `png_to_h5.py`.

  ```
  cd data/test/507727402
  python ../../../fftracer/utils/png_to_h5.py 507727402_raw.h5
  ```

  b. Run the inference step `[coming soon]`