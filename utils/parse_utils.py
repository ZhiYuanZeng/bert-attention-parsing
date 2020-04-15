import torch
import numpy as np
import nltk
import re


word_tags = set(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB'])

def get_pos_mask(max_len, scale=1.0):
    positional_mask = torch.zeros([max_len, max_len])
    for i in range(max_len):
        for j in range(max_len):
            positional_mask[i, j] = abs(j-i)
    return 1/((positional_mask)**scale+1)

def compare_heads(attentions: tuple, tree_distance: list, bpe_ids: list):
    """
    compare and visualize the distance between attention and tree distance matrix
        attentions: ([head,len,len],...)
        tree_distance: [[len,len],..]
        bpe_ids: [bpe_id,..]
    """
    # attentions = torch.stack(attentions, dim=0)
    # attentions = torch.transpose(attentions, 1, 2) # change bsz and head dimension
    
    tree_distance = -torch.tensor(tree_distance).type(attentions.type()) # [len,len]
    tree_distance=torch.softmax(tree_distance,dim=-1)
    tree_distance = tree_distance-torch.diag(tree_distance.diag()) # remove diag

    L=len(tree_distance)
    layer_count, head_count = attentions.shape[0], attentions.shape[1]
    distances = np.zeros([layer_count, head_count])
    
    # total_sequence_len = sum([len(torch.nonzero(mask))
    #                           for mask in attention_mask])

    # attention_mask=~(attention_mask.type(torch.cuda.ByteTensor)).unsqueeze(-2) # broadcast the row not the column!
    # tree_distance.masked_fill_(attention_mask,float('-inf'))
    # tree_distance = torch.softmax(tree_distance,dim=-1)

    for layer_nb in range(len(attentions)):
        for head_nb, head in enumerate(attentions[layer_nb]):
            # head: [bsz,len,len]
            # mse_distance: [bsz,len,len]
            head=AttentionScore._reduce_matrix(bpe_ids , head)
            head = head-torch.diag(head.diag()) # remove diag
            mse_distance = (head-tree_distance)**2
            mse_distance = torch.sqrt(mse_distance.sum(
                dim=(-1, -2))/(L**2))
            distances[layer_nb, head_nb] = mse_distance
    return distances


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def load_embeddings_txt(path):
  words = pd.read_csv(path, sep=" ", index_col=0,
                      na_values=None, keep_default_na=False, header=None,
                      quoting=csv.QUOTE_NONE)
  matrix = words.values
  index_to_word = list(words.index)
  word_to_index = {
    word: ind for ind, word in enumerate(index_to_word)
  }
  return matrix, word_to_index, index_to_word

def evalb(pred_tree_list, targ_tree_list):
    import os
    import subprocess
    import tempfile
    import re
    import nltk

    temp_path = tempfile.TemporaryDirectory(prefix="evalb-")
    temp_file_path = os.path.join(temp_path.name, "pred_trees.txt")
    temp_targ_path = os.path.join(temp_path.name, "true_trees.txt")
    temp_eval_path = os.path.join(temp_path.name, "evals.txt")

    print("Temp: {}, {}".format(temp_file_path, temp_targ_path))
    temp_tree_file = open(temp_file_path, "w")
    temp_targ_file = open(temp_targ_path, "w")

    for pred_tree, targ_tree in zip(pred_tree_list, targ_tree_list):
        def process_str_tree(str_tree):
            return re.sub('[ |\n]+', ' ', str_tree)

        def list2tree(node):
            if isinstance(node, list):
                tree = []
                for child in node:
                    tree.append(list2tree(child))
                return nltk.Tree('<unk>', tree)
            elif isinstance(node, str):
                return nltk.Tree('<word>', [node])

        temp_tree_file.write(process_str_tree(str(list2tree(pred_tree)).lower()) + '\n')
        temp_targ_file.write(process_str_tree(str(list2tree(targ_tree)).lower()) + '\n')

    temp_tree_file.close()
    temp_targ_file.close()

    evalb_dir = os.path.join(os.getcwd(), "EVALB")
    evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        temp_targ_path,
        temp_file_path,
        temp_eval_path)

    subprocess.run(command, shell=True)

    with open(temp_eval_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_fscore = float(match.group(1))
                break

    temp_path.cleanup()

    print('-' * 80)
    print('Evalb Prec:', evalb_precision,
          ', Evalb Reca:', evalb_recall,
          ', Evalb F1:', evalb_fscore)

    return {'prec':evalb_precision,'rec':evalb_recall,"f1":evalb_fscore}

def MRG(tr):
    if isinstance(tr, str):
        #return '(' + tr + ')'
        return tr + ' '
    else:
        s = '( '
        for subtr in tr:
            s += MRG(subtr)
        s += ') '
        return s

def MRG_labeled(tr):
    if isinstance(tr, nltk.Tree):
        if tr.label() in word_tags:
            return tr.leaves()[0] + ' '
        else:
            s = '(%s ' % (re.split(r'[-=]', tr.label())[0])
            for subtr in tr:
                s += MRG_labeled(subtr)
            s += ') '
            return s
    else:
        return ''

def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1


def comp_tree(parse_trees,tgt_trees):
    """ compare a predicted tree with target tree """
    model_out, _ = get_brackets(parse_trees)
    std_out, _ = get_brackets(tgt_trees)
    overlap = model_out.intersection(std_out)
    
    prec = float(len(overlap)) / (len(model_out) + 1e-8)
    reca = float(len(overlap)) / (len(std_out) + 1e-8)
    if len(std_out) == 0:
        reca = 1.
    if len(model_out) == 0:
        prec = 1.
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    return prec,reca,f1