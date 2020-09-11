export CUDA_VISIBLE_DEVICES=1

layer=8
output_dir=output/pred_label/unfrozen_bert
tensorboard_dir=runs/pred_label/unfrozen_bert
if [ ! -d $output_dir ]; then
    echo $output_dir not exists, mkdir
    mkdir $output_dir
fi
if [ ! -d $tensorboard_dir ]; then
    echo $tensorboard_dir not exists, mkdir
    mkdir $tensorboard_dir
fi

python model/supervised_parsing.py \
--data_dir data/ptb_supervised \
--model_name_or_path bert_pretrained/ \
--tokenizer_name bert_pretrained/ \
--config_name bert_pretrained/ \
--layer_nb $layer \
--max_seq_length 128 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 64 \
--learning_rate 5e-5 \
--logging_steps 100 \
--save_steps 200 \
--max_steps 4800 \
--dropout 0.1 \
--output_dir $output_dir \
--tensorboard_dir $tensorboard_dir \
--gradient_accumulation_steps 2 \
--task_name train \
--loss_function mle \
--is_supervised \
--do_lower_case \
--pred_label \
--evaluate_during_training