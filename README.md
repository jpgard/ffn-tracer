# ffn-tracer
Neuron tracing based on flood-filling networks (FFN)


![example neuron](./img/patch_and_label_507727402_f32.png)

Before executing the steps below, set up and activate a virtual environment using the `requirements.txt` file.

In order to train a ffn-tracer model, follow these steps:

1. Determine seed locations for each neuron. Seed locations can either be manually determined by inspecting each image using a tool such as [GIMP](https://www.gimp.org/) or Vaa3D, or by a custom algorithm Seed locations should be stored in a CSV file. For an example, see `seed_locations.csv` in this repo.

2. Create `tfrecord`s and training data coordinates for your dataset. For example, to replicate the datasets used in our analysis, run:

    ``` 
    python generate_mozak_data.py \
    --dataset_ids 319215569 515843906 541830986 397462955 \
        518298467 548268538 476667707 518358134 550168314 \
        476912429 518784828 565040416 495358721 520260582 \
        565298596 507727402 521693148 565636436 508767079 \
        521702225 565724110 508821490 522442346 570369389 \
        515548817 529751320 \
        --gs_dir data/gold_standard \
        --img_dir data/img \
        --seed_csv data/seed_locations/seed_locations.csv \
        --out_dir data/ \
        --num_training_coords 2500
    ```
    
    This deposits a set of `tfrecord`s containing the training data
    into `out_dir/tfrecords`, one `tfrecord` per dataset.
    
3. *Training*:

    Set the learning rate and depth of the model. The default ffn learning rate is 0.001 and the depth is 9.

    ``` 
    export LEARNING_RATE=0.001
    export DEPTH=9
    ```

    a. Initiate model training. You should determine values for `image_mean` and `image_stddev` for your data. Set the desired number of training iterations via `max_steps`.
    
    ```
    python train.py --tfrecord_dir ./data/tfrecords \
        --out_dir . --coordinate_dir ./data/coords \
         --image_mean 78 --image_stddev 20 \
         --train_dir ./training-logs/lr${LEARNING_RATE}depth${DEPTH} \
         --learning_rate $LEARNING_RATE \
         --max_steps 10000000 \
         --visible_gpus=0,1
    ```
    
    b. (**optional, but recommended**) initiate TensorBoard to monitor training and view sample labeled images:
    
    `tensorboard --logdir ./training-logs`
    
    Note that if you are running training on a remote server, in order to view the TensorBoard output in your browser, you will first need to run `ssh -N -f -L localhost:16006:localhost:6006 user@hostname`.

4. *Inference*: Run inference on a new dataset.

  a. Generate the target volume as hdf5. Note that the working directory should contain a set of z-slice images as `.png` files. These are assembled into the test volume by `png_to_h5.py`. For more examples of input png data see e.g. the files [here](https://github.com/janelia-flyem/neuroproof_examples/tree/master/training_sample2/grayscale_maps).

  ```
  cd data/test/507727402
  python ../../../fftracer/utils/png_to_h5.py 507727402_raw.h5
  ```
  
  b. Set up the jupyer kernel (if intending to use jupyter notebook for interence, which is recommended since it allows for manually specifying seed):
  
  ``` 
  source venv/bin/activate
  ipython kernel install --user --name=ffn
  ```
  
  After this, navigate to the location of jupyter kernels (on mac this is `/Users/yourname/Library/Jupyter/kernels/ffn`) and verify that the file `kernel.json` points to the correct PYthon interpreter (on some setups, the Python interpreter needs to be set manually). The `kernel.json` file should look like this:
  ```
      {
     "argv": [
      "path/to/repo/ffn-tracer/venv/bin/python", # line to check
      "-m",
      "ipykernel_launcher",
      "-f",
      "{connection_file}"
     ],
     "display_name": "ffn",
     "language": "python"
    }
  ```

  c. Run the inference step `[coming soon]`