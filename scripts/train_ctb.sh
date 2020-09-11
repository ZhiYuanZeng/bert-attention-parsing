export CUDA_VISIBLE_DEVICES=2

layer=8
output_dir=output/ctb/
tensorboard_dir=runs/ctb/
if [ ! -d $output_dir ]; then
    echo $output_dir not exists, mkdir
    mkdir $output_dir
fi
if [ ! -d $tensorboard_dir ]; then
    echo $tensorboard_dir not exists, mkdir
    mkdir $tensorboard_dir
fi

python model/bert_parsing.py \
--data_dir data/ptb_supervised \
--model_name_or_path bert-base-chinese \
--tokenizer_name bert-base-chinese \
--config_name bert-base-chinese \
--layer_nb $layer \
--max_seq_length 128 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--learning_rate 1e-3 \
--logging_steps 50 \
--save_steps 200 \
--max_steps 5000 \
--dropout 0.1 \
--output_dir $output_dir \
--tensorboard_dir $tensorboard_dir \
--gradient_accumulation_steps 8 \
--task_name train \
--is_supervised \
--evaluate_during_training \
--frozen_bert \
--do_lower_case \
--lang zh
# --few_shot 1700 \