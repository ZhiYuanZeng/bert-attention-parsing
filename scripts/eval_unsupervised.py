import re
import numpy as np
import subprocess

fname='/data/zyzeng/projects/bert_parsing/nohup1.zh'
lang='zh'
command = "python model/unsupervised_parsing.py \
--model_name_or_path  bert-base-multilingual-uncased \
--max_seq_length 256 \
--per_gpu_eval_batch_size 64 \
--layer_nb {} \
--head_nb {} \
--task_name val \
--decoding cky \
--multi-lingual \
--remove_bpe \
--lang {}"

def eval():
    matrix=np.zeros([12,12])
    with open(fname,'r') as f: 
        line=f.readline() 
        while line: 
            match1=re.search(r'layer nb:(\d+)',line)
            match2=re.search(r'head nb:(\d+)',line)
            if match1 and match2:
                layer_nb=match1.group(1)
                head_nb=match2.group(1)
            match3=re.search(r'Evalb F1: (.*)',line)
            if match3:
                f1=match3.group(1)
                matrix[int(layer_nb)][int(head_nb)]=f1
            line=f.readline()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j]==0:
                # print(command.format(i,j,lang))
                subprocess.run(command.format(i,j,lang),shell=True)

if __name__ == "__main__":
    eval()