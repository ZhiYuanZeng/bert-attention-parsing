#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

layer=8
dir="/data/zyzeng/projects/bert_parsing/output/few_shot/multilingual/"
# all_checkpoints="supervised/checkpoint-2000 pred_label/frozen_bert/checkpoint-2000 unfrozen_bert/checkpoint-4200 pred_label/unfrozen_bert/checkpoint-4200"
pretrained_model="bert-base-multilingual-uncased"
all_src_langs="en"
all_tgt_langs="zh"
for src_lang in $all_src_langs
do
for tgt_lang in $all_tgt_langs
do
checkpoint_path=$dir/$src_lang/checkpoint-800
echo "src language $checkpoint_path"
python model/supervised_parsing.py \
--model_name_or_path $pretrained_model \
--checkpoint_path $checkpoint_path \
--task_name test \
--do_lower_case \
--multi_lingual \
--lang $tgt_lang
done
done