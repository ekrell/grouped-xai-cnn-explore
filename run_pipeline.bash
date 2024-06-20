# Pipeline to generate NN attribution experiments

config=$1
runopts=$2

if [ -z "${config}" ]; then
  echo "Must supply config file argument!"
  echo "EX:   bash run_pipeline.bash tornado/config_cnn_classification.yaml"
  exit -1
fi

echo "Config file: ${config}"

yaml() {
  key=$1
  file=$2
  out=$(yq $key ${file} | tr -d '[]' | sed -e 's/",/"/g' | sed -e 's/"//g') 
  echo $out
}

# Task: either 'classification' or 'regression'
task=$(yaml ".task" $config)

if [ "${task}" == "classification" ]; then
  regression_opt=""
elif [ "${task}" == "regression" ]; then
  regression_opt="--regression"
else
  echo "Unrecognized task '${task}'. Expected 'classification' or 'regression'."
  exit -1
fi

# Path to '.npz' file with target rasters and predictors
data_file=$(yaml ".data_file" $config)

# Path to store all output files
out_dir=$(yaml ".out_dir" $config)/

# Name of targets variable in data file (e.g. 'y' or 'y_regr')
yname=$(yaml ".yname" $config)

# Name of model evaluation metric to use in plots (e.g. 'balanced_accuracy', 'r2')
metric_name=$(yaml ".metric_name" $config)

# Model indices for training repetitions
models=($(yaml ".models" $config))

# Samples to use for training (will use first N samples)
n_train_samples=$(yaml ".n_train_samples" $config)

# Which raster bands to include
channels=($(yaml ".channels" $config))

# Scaling factors to apply to the input rasters (along rows, channels)
img_scales=($(yaml ".img_scales" $config))

# Convolutional kernel widths (e.g. '4' means a 4x4 kernel)
filter_widths=($(yaml ".filter_widths" $config))

# Superpixel sizes (comma-delimited list e.g. '1,2,4' means 1x1, 2x2, and 4x4 patches)
patch_sizes=$(yaml ".patch_sizes" $config)

# Huperparameter: learning rate (e.g. 0.001)
learning_rate=$(yaml ".learning_rate" $config)

# hyperparameter: epochs (e.g. 50)
epochs=$(yaml ".epochs" $config)

# Control which pipeline steps to run
do_train=true
do_eval=true
do_xai=true
do_plot=true

if [ -n "${runopts}" ]; then
  runopts=$(echo "$runopts" | tr 'a-z' 'A-Z')

  declare -a a   # define array a
  for ((i=0; i<${#runopts}; i++)); do a[$i]="${runopts:$i:1}"; done

  if [ "${a[0]}" == "F" ]; then
    do_train=false
  fi
  if [ "${a[1]}" == "F" ]; then
    do_eval=false
  fi
  if [ "${a[2]}" == "F" ]; then
    do_xai=false
  fi
  if [ "${a[3]}" == "F" ]; then
    do_plot=false
  fi

  echo "Controlling pipeline with string '${runopts}':"
  echo "train.py: ${do_train}"
  echo "eval.py: ${do_eval}"
  echo "explain.py: ${do_xai}"
  echo "plot.py: ${do_plot}"
fi

for channel in "${channels[@]}"; do
  for scale in "${img_scales[@]}"; do
    for width in "${filter_widths[@]}"; do

      attrs_files=""
      metrics_files=""
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
            --filter_width ${width} \
            --y_name ${yname}   ${regression_opt} \
            --learning_rate ${learning_rate} \
            --epochs ${epochs}
        fi

        # Compute model evaluations
        metrics_file=${out_dir}/model_c-${channel}_s-${scale}_w-${width}__${model}.csv
        indices_file=${out_dir}/model_c-${channel}_s-${scale}_w-${width}__${model}.txt
        if ${do_eval}; then
          python eval.py \
            --data_file ${data_file} \
            --model_file ${model_file} \
            --num_samples ${n_train_samples} \
            --scale_data ${scale} \
            --channels ${channel} \
            --metrics_file ${metrics_file} \
            --indices_file ${indices_file} \
            --y_name ${yname}   ${regression_opt}
        fi

        # Get samples from each outcome
        samples=$(sed -n '1p' ${indices_file} | awk '{print $2}' | cut -d, -f1-10)
        samples=${samples},$(sed -n '2p' ${indices_file} | awk '{print $2}' | cut -d, -f1-10)
        samples=${samples},$(sed -n '3p' ${indices_file} | awk '{print $2}' | cut -d, -f1-10)
        samples=${samples},$(sed -n '4p' ${indices_file} | awk '{print $2}' | cut -d, -f1-10)

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
            --out_file ${attrs_file} \
            --y_name ${yname}
        fi

        # Store up a semicolon-delimited file lists
        attrs_files="${attrs_files}${attrs_file};"
        metrics_files="${metrics_files}${metrics_file};"
      done
      # Remove last comma to make a proper comma-delimited list
      attrs_files=${attrs_files::-1}
      metrics_files=${metrics_files::-1}

      # Plot XAI for the set of model repetitions
      xai_dir=${out_dir}/xai_c-${channel}_s-${scale}_w-${width}_m-occlusion/
      mkdir -p ${xai_dir}
      if ${do_plot}; then
        python plot.py \
          --attr_files ${attrs_files} \
          --metric_files ${metrics_files} \
          --out_dir "${xai_dir}/" \
          --metric ${metric_name}
      fi
    done
  done
done
