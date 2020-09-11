export CUDA_VISIBLE_DEVICES=1
layers=`seq 6 11`
echo train all layers $layers

for layer in $layers
do
    output_dir=./output/few_shot/layers/layer$layer
    tensorboard_dir=./runs/few_shot/layers/layer$layer
    if ! test -e $output_dir; then
        echo $output_dir is not exist, create $output_dir
        mkdir $output_dir
    fi
    if ! test -e $tensorboard_dir; then
        echo $tensorboard_dir is not exist, create $tensorboard_dir
        mkdir $tensorboard_dir
    fi
    echo train layer $layer
    python model/supervised_parsing.py \
    --data_dir data/ptb_supervised \
    --model_name_or_path bert_pretrained \
    --tokenizer_name bert_pretrained \
    --config_name bert_pretrained/ \
    --layer_nb $layer \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 10 \
    --per_gpu_eval_batch_size 64 \
    --learning_rate 1e-3 \
    --logging_steps 100 \
    --save_steps 100 \
    --max_steps 600 \
    --dropout 0.3 \
    --output_dir $output_dir \
    --tensorboard_dir $tensorboard_dir \
    --gradient_accumulation_steps 1 \
    --few_shot 80 \
    --task_name train \
    --is_supervised \
    --frozen_bert \
    --do_lower_case \
    --evaluate_during_training
done