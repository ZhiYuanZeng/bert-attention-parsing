export CUDA_VISIBLE_DEVICES=3
layers="5 4 3 2 1 0"
echo train all layers $layers

for layer in $layers
do
    output_dir=./output/few_shot/layer$layer
    tensorboard_dir=./runs/few_shot/layer$layer
    if ! test -e $output_dir; then
        echo $output_dir is not exist, create $output_dir
        mkdir $output_dir
    fi
    if ! test -e $tensorboard_dir; then
        echo $tensorboard_dir is not exist, create $tensorboard_dir
        mkdir $tensorboard_dir
    fi
    echo train layer $layer
    python model/bert_parsing.py \
    --data_dir data/ptb_supervised \
    --model_name_or_path bert_pretrained \
    --tokenizer_name bert_pretrained \
    --config_name bert_pretrained/ \
    --layer_nb $layer \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --learning_rate 1e-3 \
    --logging_steps 64 \
    --save_steps 100 \
    --num_train_epochs 20 \
    --dropout 0.3 \
    --output_dir $output_dir \
    --tensorboard_dir $tensorboard_dir \
    --gradient_accumulation_steps 8 \
    --task_name train \
    --few_shot 1700 \
    --supervised_parsing \
    --evaluate_during_training
done