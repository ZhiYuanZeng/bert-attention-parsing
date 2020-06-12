import sys
sys.path[0] += '/../'

from model.bert_head import BertHead
from utils.parse_utils import evalb, comp_tree
# from dataloader.unsupervised_data import load_datasets as unsupervised_load_datasets
from dataloader.supervised_data import load_datasets as supervised_load_datasets
from utils.parse_comparison import corpus_stats_labeled, corpus_average_depth
from utils.data_utils import collate_fn, en_label2idx
from utils.visualize import visual_attention,visual_hiddens

from transformers import BertConfig, BertModel, BertTokenizer,AdamW
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
# from model.bert_for_adj import BertForAdj
import os
import logging
import argparse
import re
import numpy as np
import random
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader,SequentialSampler, RandomSampler
from nltk.tree import Tree


try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def train(args, model, tokenizer, checkpoint=None):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.tensorboard_dir)
    dataset=supervised_load_datasets(args, tokenizer)
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=train_sampler)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // max(len(dataloader) // args.gradient_accumulation_steps, 1) + 1
    else:
        t_total = len(dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if checkpoint is not None:
        if isinstance(checkpoint['optimizer'],dict): optimizer.load_state_dict(checkpoint['optimizer'])
        else: optimizer=checkpoint['optimizer']
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if checkpoint is not None: scheduler.load_state_dict(checkpoint['schedule'])
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    global_step = checkpoint['step'] if checkpoint is not None else 0
    tr_loss, logging_loss, tr_acc, tr_f1 = 0.0, 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    for _ in train_iterator:
        epoch_iterator = tqdm(dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        iter_samples=0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            if args.is_supervised:
                input_ids,attention_mask,bpe_ids,labels=batch
                iter_samples+=len(input_ids)
                if args.few_shot>0 and iter_samples>args.few_shot: break # few shot training
                inputs = input_ids.to(args.device)
                loss, acc, f1 = model(inputs, attention_mask, bpe_ids, labels, inner_only=args.inner_only)
            else:
                inputs = input_ids.to(args.device)
                loss, acc = model(inputs, attention_mask)
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                acc = acc / args.gradient_accumulation_steps
                f1 = f1 / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            tr_acc += acc
            tr_f1 += f1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics TODO eval
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = eval(args, model, tokenizer, prefix=str(global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logger.info('step: {} ,loss: {}, acc: {}, f1: {}'.format(
                        global_step, (tr_loss - logging_loss)/args.logging_steps, tr_acc/args.logging_steps, tr_f1/args.logging_steps
                    ))
                    logging_loss = tr_loss
                    tr_acc=0.
                    tr_f1=0.
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_path = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    checkpoint={'key':model.key_proj.state_dict(),
                                'query':model.query_proj.state_dict(), 
                                'schedule': scheduler.state_dict(),
                                'optimizer':optimizer.state_dict(), 
                                'step': global_step, 
                                'args': args}
                    if (not args.frozen_bert) or (not args.is_supervised): checkpoint['bert']=model.bert.state_dict()
                    if hasattr(model, 'label_predictor'): checkpoint['label_predictor']=model.label_predictor.state_dict()
                    torch.save(checkpoint, output_path)
                    logger.info("Saving model checkpoint to %s", output_path)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break


        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def eval(args, model, tokenizer,prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = supervised_load_datasets(args, tokenizer, 'val')

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    avg_acc,avg_f1,eval_loss = 0.,0.,0.
    nb_eval_steps = 0
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids,attention_mask,bpe_ids,labels,_,_=batch
        inputs = input_ids.to(args.device)
        with torch.no_grad():
            if args.is_supervised:
                loss,acc,f1 = model(inputs, attention_mask,bpe_ids, labels, inner_only=args.inner_only)
            else:
                loss, acc = model(inputs, attention_mask)
            eval_loss += loss.mean().item()
            avg_acc+=acc
            avg_f1+=f1
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    avg_acc=avg_acc/nb_eval_steps
    avg_f1=avg_f1/nb_eval_steps

    result = {
        "loss": eval_loss,
        "acc": avg_acc,
        'f1': avg_f1
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

def test(args, model, tokenizer,prefix=''):
    # test parsing performance
    test_dataset = supervised_load_datasets(args, tokenizer, task=args.task_name)
    args.per_gpu_eval_batch_size=64

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running testing {} *****".format(prefix))
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    all_pred_trees=[]
    all_std_trees=[]
    for batch in tqdm(eval_dataloader, desc="Evaluate parsing"):
        input_ids,attention_mask,bpe_ids,_,trees,nltk_trees=batch
        inputs = input_ids.to(args.device)
        all_std_trees.extend(trees)
        
        with torch.no_grad():
            sents = [[tokenizer.convert_ids_to_tokens(i.item()) for i in ids[1:len(torch.nonzero(masks))-1]]
                for ids, masks in zip(inputs,attention_mask)] # detokenize
            sents=[tokenizer.convert_tokens_to_string(s).split()
                     for s in sents] # remove bpe
            if flat_tree(trees[0])!=sents[0]:
                print('error tree')
                continue
            pred_trees, all_attens, all_keys, all_querys = model.parse(
                inputs, attention_mask,bpe_ids, sents, args.rm_bpe_from_a,args.decoding, inner_only=args.inner_only)
            all_pred_trees.extend(pred_trees)
            for i,(a,s) in enumerate(zip(all_attens, sents)):
                if ' '.join(s).startswith('under an agreement signed'):
                    visual_attention([np.exp(a),],[s,],'attention-frozen.svg')
                    pass
            # visual_hiddens(query, key, sents)
            # visual_hiddens(all_querys, all_keys, sents)
            # visual_attention(all_attens, sents)
            # print(trees)
        # eval step
        f1_list=[]
        for pred_tree, std_tree in zip(pred_trees, trees):
            prec, reca, f1 = comp_tree(pred_tree, std_tree)
            f1_list.append(f1)
        print(sum(f1_list)/len(f1_list))
        nb_eval_steps += 1
    eval_res=evalb(all_pred_trees,all_std_trees) # eval all

    print(eval_res)
    checkpoint_dir='/'.join(re.split('/*',args.checkpoint_path)[:-1])
    output_eval_file = os.path.join(checkpoint_dir, "parse_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** parse results {} *****".format(prefix))
        writer.write("***** parse results {} *****".format(prefix))
        for key in sorted(eval_res.keys()):
            logger.info("  %s = %s", key, str(eval_res[key]))
            writer.write("%s = %s\n" % (key, str(eval_res[key])))

    return eval_res

def convert_to_nltk_tree(tree):
    if isinstance(tree,str):
        return tree
    tree_node=Tree(tree[0],[])
    for child in tree[1:]:
        child_node=convert_to_nltk_tree(child)
        tree_node.insert(len(tree_node),child_node)
    return tree_node

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
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
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
    checkpoint=None
    task_name=args.task_name
    if args.checkpoint_path != '':
        checkpoint=torch.load(args.checkpoint_path)
    if checkpoint is not None and checkpoint.get("bert") is not None:
        bert_model=model_class(config)
    else:
        bert_model = model_class.from_pretrained(args.model_name_or_path,
                                            config=config, 
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    if checkpoint is not None and checkpoint.get('args') is not None:
        dropout,max_steps=args.dropout,args.max_steps
        args=checkpoint['args']
        args.task_name,args.max_steps=task_name,max_steps
        args.dropout=dropout
        if not hasattr(args, 'is_supervised'): args.is_supervised=True
        if not hasattr(args, 'lang'): args.lang='en'
        if not hasattr(args, 'rm_bpe_from_a'): args.rm_bpe_from_a=False
        if not hasattr(args, 'use_bert_head'): args.use_bert_head=-1
        if not hasattr(args, 'pred_label'): args.pred_label=False
        if not hasattr(args, 'inner_only'): args.inner_only=False
        print(args.layer_nb)
    logger.info("arguments: %s", args)
    if args.is_supervised:
        if args.frozen_bert and task_name=='train':
            for param in bert_model.parameters():
                param.requires_grad_(False)

        model=BertHead(bert_model, config.hidden_size,args.head_count,args.layer_nb,args.dropout,
                    args.loss_function, args.rm_bpe_from_a,len(en_label2idx) if args.pred_label else 0)
        if checkpoint is not None:
            if checkpoint.get('key') is not None and checkpoint.get('query') is not None:
                if isinstance(args.layer_nb, int):
                    model.key_proj[0].load_state_dict(checkpoint['key'])
                    model.query_proj[0].load_state_dict(checkpoint['query'])
                else:
                    model.key_proj.load_state_dict(checkpoint['key'])
                    model.query_proj.load_state_dict(checkpoint['query'])

            if checkpoint.get('bert') is not None: model.bert.load_state_dict(checkpoint['bert'])
            if checkpoint.get('label_predictor') is not None: model.label_predictor.load_state_dict(checkpoint['label_predictor'])
    else:
        model=bert_model
    model=model.to(device)
    if task_name=='train':
        train(args, model, tokenizer, checkpoint)
    if task_name=='val':
        test(args, model, tokenizer)
    elif task_name=='test':
        test(args, model, tokenizer)

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str,help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str,help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--output_dir", default=None, type=str,help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--task_name", default=None, type=str, choices=['train','val','test'],help="train or eval or parse")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=300,help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,help="random seed for initialization")

    parser.add_argument('--tpu', action='store_true',help="Whether to run on the TPU defined in the environment variables")
    parser.add_argument('--tpu_ip_address', type=str, default='',help="TPU IP address if none are set in the environment variables")
    parser.add_argument('--tpu_name', type=str, default='',help="TPU name if none are set in the environment variables")
    parser.add_argument('--xrt_tpu_config', type=str, default='',help="XRT TPU config if none are set in the environment variables")

    parser.add_argument('--fp16', action='store_true',help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='',help="For distant debugging.")
    parser.add_argument('--server_port', type=str,
                        default='', help="For distant debugging.")
    parser.add_argument('--log_path', type=str,
                        default='', help="For distant debugging.")
    parser.add_argument('--head_nb', type=int, default='-1', help="head number for parsing")
    parser.add_argument('--wsj10', action='store_true', help="test on wsj10")
    parser.add_argument('--inner_only', action='store_true',help="whether to only compute inside score")
    parser.add_argument('--decoding', type=str, default='cky', choices=['cky','greedy'] ,help="decoding method, cky/greedy")
    parser.add_argument('--is_supervised', action='store_true', help=" supervised_train")
    parser.add_argument('--tensorboard_dir', type=str, default='runs', help="logdir of tensorboard")
    parser.add_argument('--checkpoint_path', type=str, default='', help="path of checkpoint")
    parser.add_argument('--few_shot', type=int, default=-1, help="few-shot training, training samples")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout rate")
    parser.add_argument('--frozen_bert', action='store_true', help="whther to frozen bert parameters")
    parser.add_argument('--frozen_head', action='store_true', help="whther to frozen head parameters")
    parser.add_argument('--rm_bpe_from_a', action='store_true', help="if true remove bpe from attention else hiddens")
    parser.add_argument('--embedding_type', type=str, default='bert-base', choices=['bert-base','bert-large','glove','elmo'])
    parser.add_argument('--negative_sample', type=int, default=2, help='negative sampling number')
    parser.add_argument('--head_count', type=int, default=1, help='how many heads, make sense in supervised mode')
    parser.add_argument('--lang', type=str, default='en', choices=['en','zh'], help='train/parse on ptb or ctb data')
    parser.add_argument('--use_bert_head', type=int, default=-1, help='use which bert head, default not use')
    parser.add_argument('--loss_function', type=str, default='mle', choices=['mle','hinge'] ,help='use hinge loss or mle loss(cross entropy)')
    parser.add_argument('--pred_label', action='store_true', help='whther to predict label')
    parser.add_argument('--layer_nb', nargs='+', help='use which layers')
    parser.add_argument('--early_stop', type=int, default=-1, help='patience of early stop')

    args = parser.parse_args()

    return args


def create_logger(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    if args.log_path is not None and args.log_path != '':
        file_handler = logging.FileHandler(filename=args.log_path, mode='a')
        logger.addHandler(file_handler)


if __name__ == "__main__":
    main()
