import sys
sys.path[0] += '/../'

import argparse
import numpy as np
import random
import logging
import os
import re
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from utils.parse_utils import evalb, MRG, MRG_labeled, comp_tree
from dataloader.supervised_data import load_datasets as supervised_load_datasets
from utils.parse_comparison import corpus_stats_labeled, corpus_average_depth
from utils.data_utils import collate_fn
from utils.visualize import visual_attention
from parser import parse_cyk
from model.split_score import split_score, remove_bpe_from_attention, remove_bpe_from_hiddens

logger = logging.getLogger(__name__)
file_to_print=open('unsupervised_res.txt','a')

def pos_scale(max_seq_length, W=2):
    """ make atten local by multiply atten with gasussian
    param:
        atten: (seq,seq)
    return:
        pos scale: (seq,seq) """
    mu=torch.arange(max_seq_length).view(-1,1)
    sigma=W/2
    gas=torch.arange(0,max_seq_length).repeat(max_seq_length,1)
    gas=torch.exp((-(gas-mu)**2).type(torch.float)/(2*sigma**2))
    return gas

def unsupervised_parsing(args, model, tokenizer, prefix=""):
    """ 
    return attention of every batch
        -get input
        -get output and ground truth
        -compute evaluate score
    """
    # load data
    eval_outputs_dirs = args.output_dir
    
    eval_dataset = supervised_load_datasets(args, tokenizer, args.task_name)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)
    # positional_mask = get_pos_mask(
    #     args.max_seq_length, scale=0.1).to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Eval!
    pred_tree_list, targ_tree_list, prec_list, reca_list, f1_list = [], [], [], [], []
    corpus_sys, corpus_ref = {}, {}
    sample_count = 0
    # atten_scale=pos_scale(args.max_seq_length,args.max_seq_length).to(args.device)

    for i, batch in enumerate(tqdm(eval_dataloader, desc="parsing")):
        model.eval()
        with torch.no_grad():
            input_ids, attention_mask, bpe_ids, _, tgt_trees, nltk_trees = batch
            tokens = [[tokenizer.convert_ids_to_tokens(id.item()) for id in ids[1:len(torch.nonzero(masks))-1]]
                    for ids, masks in zip(batch[0], batch[1])]
            strings = [tokenizer.convert_tokens_to_string(
                tks).split() for tks in tokens]

            inputs = {'input_ids':      input_ids,
                      'attention_mask': attention_mask,
                      }
            outputs = model(**inputs)
            _,_, hiddens, attentions = outputs  # (layer_nb,bsz,head,m,m)
            hiddens = hiddens[args.layer_nb]
            # attentions=attentions[args.layer_nb][:,args.head_nb]
            # random_layer,random_head=random.randint(0,11),random.randint(0,11)
            # attentions=attentions[random_layer][:,random_head]
            attentions=(attentions[9][:,3]+attentions[7][:,10])/2
            # attentions2 = attentions[7][:,10]
            # scores=[(s1+s2)/2 for s1,s2 in zip(scores1,scores2)]
            pred_trees=[]
            for i in range(len(hiddens)):
                seq_len=len(attention_mask[i].nonzero())-2-len(bpe_ids[i])
                h,a,s=hiddens[i],attentions[i],strings[i]
                h=remove_bpe_from_hiddens(bpe_ids[i], h)
                a=remove_bpe_from_attention(bpe_ids[i], a)
                h=h[1:1+seq_len]
                a=a[1:1+seq_len,1:1+seq_len]
                scores = split_score(h, a, bpe_ids[i], args.relevance_type, args.norm, args.inner_only)
                tree = parse_cyk(scores, s)
                pred_trees.append(tree)
                # visual attention
                # gold_tree=str(nltk_trees[i])
                # if 'ADJP' in gold_tree:
                #     print('------------ADJP-----------')
                #     m=re.findall(r'(ADJP[^\n]*)\n',str(gold_tree))
                #     print(m)
                # if 'ADVP' in gold_tree:
                #     print('------------ADVP-----------')
                #     m=re.findall(r'(ADJP[^\n]*)\n',str(gold_tree))
                #     print(m)
                # if 'ADJP' in gold_tree:
                #     visual_attention([a.cpu().numpy(),],[strings[i],],'attention-unsupervised.png')
                #     pass

            # evaluate
            for i,(pred_tree, tgt_tree, nltk_tree) in enumerate(zip(pred_trees, tgt_trees, nltk_trees)):
                prec, reca, f1 = comp_tree(pred_tree, tgt_tree)
                prec_list.append(prec)
                reca_list.append(reca)
                f1_list.append(f1)
                # if f1<0.45 and len(str(tgt_tree))<100:
                #     logger.info(f'f1:{f1}\nstd tree:{tgt_tree}\npred tree:{pred_tree}')
                corpus_sys[sample_count] = MRG(pred_tree)
                corpus_ref[sample_count] = MRG_labeled(nltk_tree)
                sample_count += 1
            pred_tree_list += pred_trees
            targ_tree_list += tgt_trees
            print(f'f1 score:{sum(f1_list)/len(f1_list)}')
    logger.info('-' * 80)
    np.set_printoptions(precision=4)
    correct, total = corpus_stats_labeled(corpus_sys, corpus_ref)
    print('-'*20 , 'model:{}---layer nb:{}---head nb:{}'.format(
        args.model_name_or_path, args.layer_nb, args.head_nb)+'-'*20)
    print('Mean Prec:', sum(prec_list)/len(prec_list),
          ', Mean Reca:', sum(reca_list)/len(reca_list),
          ', Mean F1:', sum(f1_list)/len(f1_list))
    print('Number of sentence: %i' % sample_count)
    print(correct)
    print(total)
    print('ADJP:', correct['ADJP'], total['ADJP'])
    print('NP:', correct['NP'], total['NP'])
    print('PP:', correct['PP'], total['PP'])
    print('INTJ:', correct['INTJ'], total['INTJ'])
    print(corpus_average_depth(corpus_sys))

    result = evalb(pred_tree_list, targ_tree_list)
    print(f'task:{args.task_name} model:{args.model_name_or_path}, layer nb:{args.layer_nb}, \
head nb:{args.head_nb}, seed: {args.seed}, f1:{result["f1"]}',file=file_to_print,flush=True)

def flat_tree(tree):
    if isinstance(tree,str): 
        tree=tree.lower()
        return [tree,]
    s=[]
    for child in tree:
        s.extend(flat_tree(child))
    return s

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    args = parse_args()

    # Setup CUDA, GPU & distributed training
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    set_seed(args)
    # Setup logging
    create_logger(args)
    config_class, model_class, tokenizer_class = BertConfig, BertModel, BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None,
                                                tokenize_chinese_chars=False,
                                                split_puntc=False
                                                )
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=0,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.output_attentions=True
    config.output_hidden_states=True
    # load model from checkpoint and pretrained model
    model = model_class.from_pretrained(args.model_name_or_path,
                                        config=config, 
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model=model.to(device)
    if os.path.isdir(args.model_name_or_path):
        args_path=os.path.join(args.model_name_or_path,'training_args.bin')
        if os.path.exists(args_path):
            train_args=torch.load(args_path)
            args.seed=train_args
    unsupervised_parsing(args, model, tokenizer)

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--task_name", default=None, type=str, choices=['train','val','test','wsj10'], required=True,
                        help="train or eval or parse")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=300,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--tpu', action='store_true',
                        help="Whether to run on the TPU defined in the environment variables")
    parser.add_argument('--tpu_ip_address', type=str, default='',
                        help="TPU IP address if none are set in the environment variables")
    parser.add_argument('--tpu_name', type=str, default='',
                        help="TPU name if none are set in the environment variables")
    parser.add_argument('--xrt_tpu_config', type=str, default='',
                        help="XRT TPU config if none are set in the environment variables")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='',
                        help="For distant debugging.")
    parser.add_argument('--server_port', type=str,
                        default='', help="For distant debugging.")
    parser.add_argument('--log_path', type=str,
                        default='', help="For distant debugging.")
    parser.add_argument('--layer_nb', type=int, default='-1', help="layer number for parsing")
    parser.add_argument('--head_nb', type=int, default='-1', help="head number for parsing")
    parser.add_argument('--relevance_type', type=str, default='attention', choices=['L2','attention'] ,help="relevance score type")
    parser.add_argument('--decoding', type=str, default='cky', choices=['cky','greedy'] ,help="decoding method, cky/greedy")
    parser.add_argument('--tensorboard_dir', type=str, default='runs', help="logdir of tensorboard")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout rate")
    parser.add_argument('--lang', type=str, default='en', choices=['en','zh'], help='train/parse on ptb or ctb data')
    parser.add_argument('--norm', action='store_true', help='whether to normalize the split scores') 
    parser.add_argument('--inner_only', action='store_true', help='whether to only use inner relevance score') 

    args = parser.parse_args()

    return args


def create_logger(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    if args.log_path is not None and args.log_path!='':
        file_handler = logging.FileHandler(filename=args.log_path, mode='w')
        logger.addHandler(file_handler)


if __name__ == "__main__":
    main()
    file_to_print.close()
