from torch.utils.data import Dataset
import torch
import collections
import os
import logging
logger = logging.getLogger(__name__)


word_tags = {
    'en':set(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']),
    'zh':set(['AD','AS','BA','CC','CD','CS''DEC','DEG','DER','DEV',
            'DT','ETC','FW','IJ','JJ','LB','LC','M','MSP','NN','NR','NT',
            'OD','ON','P','PN','SB','SP','VA','VC','VE','VV'
    ])
}

ctb_dir='/home/zyzeng/nltk_data/corpora/ctb/cleaned/'

def load_reader_and_filedids(lang):
    assert lang in ('en','zh')
    train_file_ids = []
    valid_file_ids = []
    test_file_ids = []
    if lang=='en':
        from nltk.corpus import ptb
        reader=ptb
        for id in reader.fileids():
            if 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/24/WSJ_2499.MRG':
                train_file_ids.append(id)
            if 'WSJ/22/WSJ_2200.MRG' <= id <= 'WSJ/22/WSJ_2299.MRG':
                valid_file_ids.append(id)
            if 'WSJ/23/WSJ_2300.MRG' <= id <= 'WSJ/23/WSJ_2399.MRG':
                test_file_ids.append(id)
        return ptb, train_file_ids, valid_file_ids, test_file_ids
    else:
        from nltk.corpus import BracketParseCorpusReader
        reader=BracketParseCorpusReader(ctb_dir, '.*')
        for id in reader.fileids():
            if id.startswith('train'):
                train_file_ids.append(id)
            elif id.startswith('val'):
                valid_file_ids.append(id)
            elif id.startswith('test'):
                test_file_ids.append(id)
    return reader, train_file_ids, valid_file_ids, test_file_ids

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
    elif isinstance(batch[0], collections.Sequence):
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

def load_glove(glove_path:str)->dict:
    glove_vectors=dict()
    with open(glove_path,'r') as f:
        line=f.readline()
        list_=line.split()
        assert len(list)==301
        token=list_[0]
        vector=torch.tensor(list_[1:])
        glove_vectors[token]=vector
    return glove_vectors