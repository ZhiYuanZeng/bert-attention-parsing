export CUDA_VISIBLE_DEVICES=2

layer=8
few_shot=-1
all_loss_func="mle"

all_seeds="4"
for seed in $all_seeds
do
    echo "###seed:$seed###"
    output_dir=./output/supervised/$seed
    tensorboard_dir=./runs/supervised/$seed
    if ! test -e $output_dir; then
        echo $output_dir is not exist, create $output_dir
        mkdir $output_dir
    fi
    if ! test -e $tensorboard_dir; then
        echo $tensorboard_dir is not exist, create $tensorboard_dir
        mkdir $tensorboard_dir
    fi
    echo train few_shot $few_shot
    python model/supervised_parsing.py \
    --data_dir data/ptb_supervised \
    --model_name_or_path bert_pretrained \
    --tokenizer_name bert_pretrained \
    --config_name bert_pretrained/ \
    --layer_nb $layer \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --learning_rate 1e-3 \
    --logging_steps 600\
    --save_steps 200 \
    --max_steps 2400 \
    --output_dir $output_dir \
    --tensorboard_dir $tensorboard_dir \
    --gradient_accumulation_steps 2 \
    --task_name train \
    --is_supervised \
    --do_lower_case \
    --frozen_bert \
    --seed $seed \
    --evaluate_during_training
done