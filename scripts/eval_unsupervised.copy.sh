export CUDA_VISIBLE_DEVICES=0
all_layers=`seq 0 11`
all_heads=`seq 0 11`
all_langs="il sp"
layer_nb=0
head_nb=0
pretrained_dir="/home/zyzeng/projects/transformers/output/pretrained/"
for lang in $all_langs
do
# echo "#######################$lang############################"
# for layer_nb in $all_layers
# do
# for head_nb in $all_heads
# do
python model/unsupervised_parsing.copy.py \
--model_name_or_path $pretrained_dir/$lang \
--max_seq_length 256 \
--per_gpu_eval_batch_size 32 \
--layer_nb $layer_nb \
--head_nb $head_nb \
--task_name test \
--decoding cky \
--remove_bpe \
--do_lower_case \
--multi-lingual \
--lang $lang
# done
# done
done
