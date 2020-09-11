export CUDA_VISIBLE_DEVICES=0

few_shot=80
all_lang='en zh ca de fr il jp sp sw'
for lang in $all_lang
do
output_dir=./output/few_shot/multilingual/$lang
tensorboard_dir=./runs/few_shot/multilingual/$lang
if ! test -e $output_dir; then   
    echo $output_dir is not exist, create $output_dir
    mkdir -p $output_dir
fi
if ! test -e $tensorboard_dir; then
    echo $tensorboard_dir is not exist, create $tensorboard_dir
    mkdir -p $tensorboard_dir
fi
echo "#####################$lang#####################"
python model/supervised_parsing.py \
--model_name_or_path bert-base-multilingual-uncased \
--layer_nb 8 \
--max_seq_length 256 \
--per_gpu_train_batch_size 10 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-5 \
--logging_steps 50 \
--save_steps 50 \
--max_steps 800 \
--dropout 0.3 \
--output_dir $output_dir \
--tensorboard_dir $tensorboard_dir \
--gradient_accumulation_steps 1 \
--task_name train \
--few_shot $few_shot \
--is_supervised \
--do_lower_case \
--lang $lang \
--multi_lingual \
--evaluate_during_training
done