export CUDA_VISIBLE_DEVICES=1

layer=12
output_dir=output/bert_large/
tensorboard_dir=runs/bert_large/
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
--model_name_or_path bert_pretrained/bert-large-uncased-pytorch_model.bin \
--tokenizer_name bert-large-uncased \
--config_name bert-large-uncased \
--layer_nb $layer \
--max_seq_length 128 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--learning_rate 5e-4 \
--logging_steps 50 \
--save_steps 200 \
--num_train_epochs 100 \
--dropout 0.1 \
--output_dir $output_dir \
--tensorboard_dir $tensorboard_dir \
--gradient_accumulation_steps 8 \
--task_name train \
--supervised_train \
--evaluate_during_training \
--frozen_bert \
# --few_shot 1700 \