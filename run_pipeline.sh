# Pipeline to generate CNN attribution experiments

# Input channels to include
channels=("0,1,2")
# Scale input (e.g. '2' doubles raster rows, cols via quadratic interpolation)
img_scales=(1 2 3)
# Convolutional filter sizes (e.g '4' means a 4x4 filter)
filter_widths=(4 8 12)
# Number of samples for training the models
n_train_samples=10000
# Path to raster dataset (shape: samples, rows, cols, bands)
data_file="tornado/data/dataset.npz"
# Path to save output models and explanations
out_dir="tornado/models/"
# Define the model training replications
models=(0 1 2 3 4 5)
# List of superpixel patch sizes
patch_sizes="1,2,4,6,8"
# List of samples for local attributions
samples=$(seq -s, 0 25)
mkdir -p ${out_dir}

# Control which pipeline steps to run
do_train=false
do_xai=true
do_plot=true

# Run XAI
for channel in "${channels[@]}"; do
  for scale in "${img_scales[@]}"; do
    for width in "${filter_widths[@]}"; do

      attrs_files=""
      for model in "${models[@]}"; do
        
        # Train model
        model_file=${out_dir}/model_c-${channel}_s-${scale}_w-${width}__${model}.keras
        if ${do_train}; then
          python train.py \
            --data_file ${data_file} \
            --model_file ${model_file} \
            --num_samples ${n_train_samples} \
            --scale_data ${scale} \
            --channels ${channel} \
            --filter_width ${width}
        fi

        # Compute model attributions
        attrs_file=${out_dir}/attrs_c-${channel}_s-${scale}_w-${width}__${model}.npz
        if ${do_xai}; then
          python explain.py \
            --data_file ${data_file} \
            --model_file ${model_file} \
            --patch_sizes ${patch_sizes} \
            --scale_data ${scale} \
            --channels ${channel} \
            --sample_idxs ${samples} \
            --out_file ${attrs_file}
        fi

        # Store up a comma-delimited list of attributions files
        attrs_files=${attrs_files}${attrs_file},
      done
      # Remove last comma to make a proper comma-delimited list
      attrs_files=${attrs_files::-1}

      # Plot XAI for the set of model repetitions
      xai_dir=${out_dir}/xai_c-${channel}_s-${scale}_w-${width}_m-occlusion/
      mkdir -p ${xai_dir}
      if ${do_plot}; then
        python plot.py \
          --attr_files ${attrs_files} \
          --out_dir "${xai_dir}/"
      fi
    done
  done
done
