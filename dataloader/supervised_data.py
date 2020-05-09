import torch
from nltk.tree import Tree
import re
from random import shuffle
from transformers import DataProcessor, InputExample, InputFeatures
import json
if __name__ == "__main__":
    import sys
    sys.path[0]+='/../'

from utils.data_utils import (word_tags,
    load_reader_and_filedids,
    load_and_cache_examples,
    PtbDataset,
    en_label2idx as label2idx
)
MAX_NUM=-1

# label2idx={'empty':0}
def get_splits(tree, start_idx ,split_list):
    global null_labels, all_labels
    if not isinstance(tree,Tree): 
        return 1
    if len(tree)==1:
        return get_splits(tree[0], start_idx, split_list)
    l=get_splits(tree[0],start_idx, split_list)
    r=get_splits(tree[1], start_idx+l, split_list)
    if '|' in tree.label():
        label='empty'
    else:
        label=re.split(r'[-=]',tree.label())[0]
    # if label2idx.get(label) is None: label2idx[label]=0
    # else: label2idx[label]+=1
    split_list.append((label2idx[label], # if '|' in label, the node is n-nary(n>2) node
                    (start_idx, start_idx+l+r-1, start_idx+l-1,),))
    return l+r

class PtbInputExample(InputExample):
    def __init__(self, guid, text_a, text_b=None, label=None, list_tree=None, nltk_tree=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.list_tree=list_tree
        self.nltk_tree=nltk_tree

class PtbProcessor(DataProcessor):
    def __init__(self, lang):
        self.lang=lang
        self.reader,self.train_file_ids,self.valid_file_ids, self.test_file_ids=load_reader_and_filedids(lang)

    def get_wsj10_examples(self):
        return self._create_examples(self.train_file_ids+self.valid_file_ids+self.test_file_ids, 'wsj10')

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self.train_file_ids, 'train')

    def get_val_examples(self):
        """See base class."""
        return self._create_examples(self.valid_file_ids, 'val')

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(self.test_file_ids, 'test')

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            if tag in word_tags[self.lang]:
                # w = w.lower()
                w = re.sub('[0-9]+', 'n', w)
                words.append(w)
        return words

    def _create_examples(self, file_ids, set_type):
        """Creates examples for the training and dev sets."""
        def tree2list(tree):
            if isinstance(tree, Tree):
                if tree.label() in word_tags[self.lang]:
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
                    else:
                        return []
            return []

        examples = []
        for i, id in enumerate(file_ids):
            if MAX_NUM>0 and i>MAX_NUM: break

            sentences = self.reader.parsed_sents(id)
            for j, sen_tree in enumerate(sentences):
                words = self.filter_words(sen_tree)  # extract words from tree
                if set_type=='wsj10' and len(words) > 10:
                    continue
                texta = ' '.join(words)
                guid = "%s-%s" % (set_type, i)
                list_tree,nltk_tree,label=None, None, None
                # sen_tree=self._remove_puntc(sen_tree) # remove puntc
                if set_type in ('test','wsj10','val'):
                    tree = tree2list(sen_tree)
                    if len(tree)==0: continue
                    list_tree, nltk_tree=tree,sen_tree
                if set_type in ('train','val'):
                    split_list=[]
                    sen_tree=self._remove_puntc(sen_tree) # remove puntc
                    if sen_tree is None or len(sen_tree)==0: continue # empty tree
                    if len(sen_tree.leaves())!=len(words):
                        print('error tree')
                        continue
                    sen_tree.chomsky_normal_form()
                    get_splits(sen_tree,0,split_list)
                    assert len(sen_tree.leaves())==(len(split_list)+1)
                    label=split_list

                examples.append(
                    PtbInputExample(guid=guid, text_a=texta, label=label, 
                        list_tree=list_tree ,nltk_tree=nltk_tree)
                )
        json.dump(label2idx,open(f'{self.lang}-label2idx.json','w'))
        return examples
    
    def _remove_puntc(self,tree):
        if len(tree)==1 and isinstance(tree[0],str) and tree.label() not in word_tags[self.lang]:
            return None
        new_t=Tree(tree.label(),[])
        for subtree in tree:
            if isinstance(subtree,Tree):
                subtree=self._remove_puntc(subtree)
            if subtree is not None and len(subtree)>0:
                new_t.insert(len(new_t),subtree)
        return new_t

    def _remove_tag(self,tree):
        if isinstance(tree, str):
            return re.sub('[0-9]+', 'n', tree.lower())
        if len(tree)==1:
            return self._remove_tag(tree[0])
        new_t=Tree(tree.label(),[])
        for subtree in tree:
            if isinstance(subtree,Tree):
                subtree=self._remove_tag(subtree)
                new_t.insert(len(new_t),subtree)
        return new_t
class PtbInputFeatures(InputFeatures):
    def __init__(self, input_ids, attention_mask, bpe_ids=None, label=None, 
                list_tree=None, nltk_tree=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.bpe_ids = bpe_ids,
        self.label = label
        self.list_tree=list_tree
        self.nltk_tree=nltk_tree

def select_list_by_indices(list_,indices):
    return [list_[i] for i in indices]

def load_datasets(args, tokenizer=None, task='train'):
    lang=args.lang
    processor=PtbProcessor(lang)
    features=load_and_cache_examples(args, processor, PtbInputFeatures, tokenizer, task, lang)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long, device=args.device)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long, device=args.device)
    all_bpe_ids = [f.bpe_ids[0] for f in features]
    all_labels = [f.label for f in features]

    if task == 'train': 
        # shuffle data
        if args.few_shot>0 and args.is_supervised: # few shot data
            shuffled_indices=list(range(args.few_shot))
        else:
            shuffled_indices=list(range(len(all_input_ids)))
        shuffle(shuffled_indices)
        all_input_ids=select_list_by_indices(all_input_ids,shuffled_indices)
        all_attention_mask=select_list_by_indices(all_attention_mask,shuffled_indices)
        all_bpe_ids=select_list_by_indices(all_bpe_ids,shuffled_indices)
        all_labels=select_list_by_indices(all_labels,shuffled_indices)
        dataset=PtbDataset(all_input_ids, all_attention_mask,all_bpe_ids, all_labels)
    else:
        all_list_trees=[f.list_tree for f in features]
        all_nltk_trees=[f.nltk_tree for f in features]
        dataset=PtbDataset(all_input_ids, all_attention_mask,all_bpe_ids,all_labels,
                            all_list_trees, all_nltk_trees)

    return dataset

if __name__ == "__main__":
    processor=PtbProcessor('en')
    test=processor.get_test_examples()
    print(label2idx)
    # test=sorted(test, key=lambda item: len(item.text_a.split()))
    pass
    # print(null_labels/all_labels)
    # train=processor.get_train_examples()
    # print(len(train))
    # wsj10=processor.get_wsj10_examples()