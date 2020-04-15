import os,logging
import torch
from transformers import DataProcessor, InputExample, InputFeatures
logger = logging.getLogger(__name__)

class PenProcessor(DataProcessor):
    """Processor for the Peen datasets.load data to examples(list of dictionary)"""
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, dir_name):
        """See base class."""
        return self._create_examples(os.path.join(dir_name, "train.txt"), "train")

    def get_dev_examples(self, dir_name):
        """See base class."""
        return self._create_examples(os.path.join(dir_name, "valid.txt"), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, path, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(path, 'r') as f:
            for (i, line) in enumerate(f):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
        return examples


def convert_examples_to_features(args, examples, tokenizer,
                                 max_length=512,
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

        features.append(
            InputFeatures(input_ids=torch.tensor(input_ids,dtype=torch.long,device=args.device),
                          attention_mask=torch.tensor(attention_mask,dtype=torch.long,device=args.device),
                          token_type_ids=None,
                          label=None
                          ))

    return features
