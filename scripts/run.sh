# /bin/bash
CUDA_VISIBLE_DEVICES=0,1
max=11
layer_nb=7
head_nb=-1
fname="bert_pretrained/checkpoint-2000"

for method in "inner"
do
    for head_nb in $(seq 6 11)
    do
        python /home/zyzeng/projects/bert_parsing/run_scripts/bert_parsing.py \
            --model_name_or_path=$fname \
            --layer_nb=$layer_nb \
            --head_nb=$head_nb \
            --method=$method \
            --do_eval
    done
done

