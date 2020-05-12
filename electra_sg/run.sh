#!/bin/bash
bucket_name="squad_c"
gs_data_root="gs://"$bucket_name
model_size="large"
model_name="electra_"$model_size
data_dir=$gs_data_root
features_dir="electra_features_sg"
features_dir=${gs_data_root}"/"${features_dir}
output_dir="electra_output_sg"
output_dir=${gs_data_root}"/"${output_dir}
rm -rf electra_squad
git clone https://github.com/timbereye/electra_squad.git
cd electra_squad/electra_sg/
python3 run_finetuning.py \
        --data-dir ${data_dir} \
        --model-name ${model_name} \
        --output-dir ${output_dir} \
        --features-dir ${features_dir} \
        --hparams '{"do_train":false,"model_size":"large","task_names":["squad"],"use_tpu":true,"tpu_name":"cx","num_tpu_cores":8,"train_batch_size":32, "num_trials":5}'