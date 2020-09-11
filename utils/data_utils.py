from torch.utils.data import Dataset
import torch
from collections import Sequence, defaultdict
import os
import random
import logging
import unicodedata
from nltk.corpus import BracketParseCorpusReader
logger = logging.getLogger(__name__)


word_tags = {
    'en':set(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']),
    'zh':set(['AD','AS','BA','CC','CD','CS''DEC','DEG','DER','DEV',
            'DT','ETC','FW','IJ','JJ','LB','LC','M','MSP','NN','NR','NT',
            'OD','ON','P','PN','SB','SP','VA','VC','VE','VV'
    ]),
    'jp':set(['QUOT','-LRB-','-RRB-','ADJI','ADJN','ADV','AX','AXD','CL','CONJ','D','FW','INTJ',
            'MD','N','NEG','NPR','NUM','P','PASS','PNL','PRO','Q','QN','SYM','VB','VB0','VB2','WADV','WD','WNUM','WPRO',]),
    'fr':set(['N','A','Adv','P','D','CL','ET','C','I','PRO','V']),
    'il':set(['N','NS','NPR','NPRS','ADJP','ADJR','ADJS','PRO','D','NUM','VB','BE','DO','HV','MD','RD','VAN','BAN',
        'DAN','HAN','VBN','BEN','DON','HVN','RDN','P','ADV','ADVR','ADVS']),
}
en_label2idx=defaultdict(int, {"empty": 1})
en_idx2label=defaultdict(str, {1: "empty"})

treebank_dir='/data/zyzeng/datasets/treebank'
train_data_size,val_data_size,test_data_size=80,1000,2000

def load_reader_and_filedids(lang,data_type):
    assert data_type in ('train','val','test')
    def filter_trees(tree, data_type):
        def _is_control(char):
            """Checks whether `chars` is a control character."""
            # These are technically control characters but we count them as whitespace
            # characters.
            if char == "\t" or char == "\n" or char == "\r":
                return False
            cat = unicodedata.category(char)
            if cat.startswith("C"):
                return True
            return False
        
        sent=tree.leaves()
        if data_type=='wsj' and len(sent)>10: return False
        if data_type!='wsj' and len(sent)>128: return False
        try:
            for c in ' '.join(sent):
                cp=ord(c)
                if cp == 0 or cp == 0xfffd or _is_control(c):
                    return False
            return True
        except:
            return False

    def filt_id(fileids,lang):
        assert lang in ('en','fr','zh')
        train_file_ids,valid_file_ids,test_file_ids=[],[],[]
        for id in fileids:
            prefix=id.split('.')[0]
            if lang=='en':
                if 'WSJ/22/WSJ_2200' <= prefix <= 'WSJ/22/WSJ_2299':
                    valid_file_ids.append(id)
                elif 'WSJ/23/WSJ_2300' <= prefix <= 'WSJ/23/WSJ_2399':
                    test_file_ids.append(id)
                else:
                    train_file_ids.append(id)        
            elif lang=='zh':
                if '0886' <= prefix <= '0931' or '1148' <= prefix <= '1151':
                    valid_file_ids.append(id)
                elif '0816' <= prefix <= '0885' or '1137' <= prefix <='1147':
                    test_file_ids.append(id)
                else:
                    train_file_ids.append(id)        
            else:
                if prefix in ('flmf3_12500_12999co','flmf7ab2ep','flmf7ad1co','flmf7ae1ep'):
                    valid_file_ids.append(id) 
                elif prefix in ('flmf3_12000_12499ep','flmf7aa1ep','flmf7aa2ep','flmf7ab1co'):
                    test_file_ids.append(id)
                else:
                    train_file_ids.append(id)
        return train_file_ids,valid_file_ids,test_file_ids

    assert lang in ('en','zh','fr','il','jp','sp','ca','sw','de')
    lang_dir=treebank_dir+'/'+lang
    reader=BracketParseCorpusReader(lang_dir, '.*')
    fileids=reader.fileids()
    if data_type=='wsj10':
        return [t for t in reader.parsed_sents(fileids) if filter_trees(t,data_type)]
    train_file_ids = []
    valid_file_ids = []
    test_file_ids = []
    if lang in ('en','zh','fr'):
        train_file_ids,valid_file_ids,test_file_ids=filt_id(fileids,lang)
        train_trees=reader.parsed_sents(train_file_ids)
        val_trees=reader.parsed_sents(valid_file_ids)
        test_trees=reader.parsed_sents(test_file_ids)
    else:
        for fid in fileids:
            if 'train' in fid:
                train_trees=reader.parsed_sents(fid)
            elif 'val' in fid:
                val_trees=reader.parsed_sents(fid)
            elif 'test' in fid:
                test_trees=reader.parsed_sents(fid)
    if data_type=='train':
        train_trees=[t for t in train_trees if filter_trees(t,data_type)]
        print(f'train:{len(train_trees)}')
        return train_trees
    elif data_type=='val':
        val_trees=[t for t in val_trees if filter_trees(t,data_type)]
        print(f'val:{len(val_trees)}')
        return val_trees
    else:
        test_trees=[t for t in test_trees if filter_trees(t,data_type)]
        print(f'test:{len(test_trees)}')
        return test_trees     

class PtbDataset(Dataset):
    def __init__(self, *objs):
        assert all(len(objs[0]) == len(obj) for obj in objs)
        self.objs = objs

    def __getitem__(self, index):
        return [obj[index] for obj in self.objs]

    def __len__(self):
        return len(self.objs[0])


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    if isinstance(batch[0], torch.Tensor):
        out = None
        return torch.stack(batch, 0, out=out)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        results = []
        for samples in transposed:
            samples = list(samples)
            if isinstance(samples[0], torch.Tensor):  # tensor
                results.append(torch.stack(samples, 0, out=None))
            else:  # list
                results.append(samples)

    return results

def get_bpe_ids(id2word, input_ids):
    bpe_ids = []
    for i, id in enumerate(input_ids):
        if id2word.get(id) is not None and '##' in id2word.get(id):
            bpe_ids.append(i)
    return bpe_ids


def convert_examples_to_features(args, feature_class, examples, tokenizer,
                                 max_length=512,
                                 label_list=None,
                                 output_mode=None,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # if len(tokenizer.decode(input_ids).split()) != len(example.text_a.split())+2: 
        #     print('error at data_utils: tokenizing error')
        #     continue
        bpe_ids = get_bpe_ids(tokenizer.ids_to_tokens, input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + \
            ([0 if mask_padding_with_zero else 1] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" %
                        " ".join([str(x) for x in attention_mask]))

        example_attrs=example.__dict__
        example_attrs.pop('guid')
        example_attrs.pop('text_a')
        example_attrs.pop('text_b')
        features.append(
            feature_class(input_ids=input_ids,
                             attention_mask=attention_mask,
                             bpe_ids=bpe_ids,
                             **example_attrs
                             ))
                             

    return features


def load_and_cache_examples(args, processor ,feature_class, tokenizer=None, task="train", lang='en'):
    """ 
    read examples using processor
    convert examples to features 
    convert to features to tensor
    make dataset
    """
    # load datasets
    assert task in ['train','val','test','wsj10']
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}'.format(task,lang))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # create datasets
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if task=='train':
            examples = processor.get_train_examples()
        elif task=='val':
            examples = processor.get_val_examples()
        elif task=='test':
            examples = processor.get_test_examples()
        else:
            examples = processor.get_wsj10_examples()
        features = convert_examples_to_features(args, feature_class, examples,
                                                tokenizer,
                                                max_length=args.max_seq_length,
                                                pad_token=tokenizer.convert_tokens_to_ids(
                                                    [tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)
    return features