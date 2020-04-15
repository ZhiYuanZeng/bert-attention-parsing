import logging
import re
import nltk
import logging
import torch
import numpy as np
from nltk.corpus import ptb
from transformers import DataProcessor, InputExample, InputFeatures
from utils.data_utils import (word_tags,
    load_reader_and_filedids,
    PtbDataset,
    load_and_cache_examples
)
logger = logging.getLogger(__name__)


    # elif 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/01/WSJ_0199.MRG' or 'WSJ/24/WSJ_2400.MRG' <= id <= 'WSJ/24/WSJ_2499.MRG':
    #     rest_file_ids.append(id)


class PtbInputExample(InputExample):
    def __init__(self, guid, text_a, text_b=None, tree=None, nltk_tree=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.tree = tree
        self.nltk_tree = nltk_tree


class PtbProcessor(DataProcessor):
    """Processor for the Peen datasets.load data to examples(list of dictionary)"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_wsj10_examples(self):
        return self._create_examples(train_file_ids, 'wsj10', wsj10=True)

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(train_file_ids, 'train')

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(valid_file_ids, 'dev')

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(test_file_ids, 'test')

    def get_labels(self):
        """See base class."""
        return [None]

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            if tag in word_tags:
                w = w.lower()
                w = re.sub('[0-9]+', 'n', w)
                words.append(w)
        return words

    def _create_examples(self, file_ids, set_type, wsj10=False):
        """Creates examples for the training and dev sets."""
        def tree2list(tree):
            if isinstance(tree, nltk.Tree):
                if tree.label() in word_tags:
                    w = tree.leaves()[0].lower()
                    w = re.sub('[0-9]+', 'n', w)
                    return w
                else:
                    root = []
                    for child in tree:
                        c = tree2list(child)
                        if c != []:
                            root.append(c)
                    if len(root) > 1:
                        return root
                    elif len(root) == 1:
                        return root[0]
            return []

        examples = []
        for i, id in enumerate(file_ids):
            sentences = ptb.parsed_sents(id)
            for sen_tree in sentences:
                words = self.filter_words(sen_tree)  # extract words from tree
                if wsj10 and len(words) > 10:
                    continue
                if set_type in ('test','wsj10'):
                    tree = tree2list(sen_tree)
                    nltk_tree=sen_tree
                else:
                    tree,nltk_tree=None,None
                    
                texta = ' '.join(words)
                guid = "%s-%s" % (set_type, i)
                examples.append(
                    PtbInputExample(guid=guid, text_a=texta, text_b=None,
                                    tree=tree, nltk_tree=sen_tree)
                )
        return examples

class PtbInputFeatures(InputFeatures):
    def __init__(self, input_ids, attention_mask, bpe_ids=None, tree=None, nltk_tree=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.bpe_ids = bpe_ids,
        self.tree = tree
        self.nltk_tree = nltk_tree

def load_datasets(args, tokenizer=None, task='train',lang='en'):
    processor=PtbProcessor(lang)
    features=load_and_cache_examples(args,processor,PtbInputFeatures ,tokenizer, task)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long, device=args.device)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long, device=args.device)
    all_bpe_ids = [f.bpe_ids[0] for f in features]
    all_trees = [f.tree for f in features]
    all_nltk_trees = [f.nltk_tree for f in features]

    dataset = PtbDataset(all_input_ids, all_attention_mask,
                         all_bpe_ids, all_trees, all_nltk_trees)

    return dataset


def tree2dis(tree_list):
    def dfs(tree_list):
        if not isinstance(tree_list, list):
            return {'count': 1}
        if len(tree_list) > 2:
            tree_list = [tree_list[:-1], tree_list[-1]]

        left, right = tree_list
        node = dict()
        left_node = dfs(left)
        right_node = dfs(right)
        node['count'] = left_node['count']+right_node['count']
        node['left'] = left_node
        node['right'] = right_node
        return node

    def t2d(node, dis, i, j):
        if node['count'] > 1:
            lc = node['left']['count']
            dis[i:(j+1), :] += 1
            dis[:, i:(j+1)] += 1
            dis[i:(j+1), i:(j+1)] -= 1
            dis[i:(lc+i), i:(lc+i)] = 0
            dis[(lc+i):(j+1), (lc+i):(j+1)] = 0
            t2d(node['left'], dis, i, lc+i-1)
            t2d(node['right'], dis, lc+i, j)

    root = dfs(tree_list)
    dis = np.zeros((root['count'], root['count']))

    t2d(root, dis, 0, root['count']-1)
    return dis


if __name__ == "__main__":
    t = [1, 2, [3, 4, [5, 6], 7, [8, [9, 10]]]]
    dis = tree2dis(t)
    print(dis)
