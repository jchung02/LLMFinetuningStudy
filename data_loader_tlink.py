import os
import copy
import json
import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.utils.class_weight import compute_class_weight as ccw

from utils import get_labels
import pdb

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, words, label):
        self.guid = guid
        self.words = words
        #self.arg1 = arg1
        #self.arg2 = arg2
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, entity_starts, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.entity_starts = entity_starts
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TlinkRE(object):
    def __init__(self, args):
        self.args = args
        self.labels_lst = get_labels(args)

    @classmethod
    def _read_file(cls, input_file):
        """Read tsv file, and return words and label as list"""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            words, label = data.split('\t')
            label = label.strip()
            words = words.split()
            #b1 = int(b1); e1 = int(e1); b2 = int(b2); e2 = int(e2)
            
            guid = "%s-%s" % (set_type, i)

            label_idx = self.labels_lst.index(label) if label in self.labels_lst else self.labels_lst.index("UNK")

            if i % 10000 == 0:
                logger.info(data)

            #examples.append(InputExample(guid=guid, words=words, arg1=(b1, e1), arg2=(b2, e2), label=label_idx))
            examples.append(InputExample(guid=guid, words=words, label=label_idx))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args["train_file"]
        elif mode == 'dev':
            file_to_read = self.args["val_file"]
        elif mode == 'test':
            file_to_read = self.args["test_file"]

        logger.info("LOOKING AT {}".format(os.path.join(self.args["data_dir"], file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args["data_dir"], file_to_read)), mode)

processors = {
    "tlink-re": TlinkRE,
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id
        
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = []
        label_ids = []

        words = example.words

        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        entity_starts = [None, None]
        for i, t in enumerate(tokens):
            if t == '[B1]':
                entity_starts[0] = i
            elif t == '[B2]':
                entity_starts[1] = i
        if None in entity_starts:
            print("Invalid entity_starts")
            exit()

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        label = example.label
        
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: {}".format(example.label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          entity_starts=entity_starts,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label
                          ))
    return features


def load_and_cache_examples(args, tokenizer, mode, use_cache=True, compute_class_weight=False):
    processor = processors[args["task"]](args)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        args["task"], list(filter(None, args["model_name_or_path"].split("/"))).pop(), args["max_seq_len"], mode)

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    cached_features_file = os.path.join(args["data_dir"], cached_file_name)
    if os.path.exists(cached_features_file) and use_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args["data_dir"])
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, args["max_seq_len"], tokenizer, pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    all_entity_starts = torch.tensor([f.entity_starts for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids,
            all_entity_starts)

    if compute_class_weight:
        all_label_ids_cpu = all_label_ids.cpu().numpy()
        cw = ccw(class_weight='balanced', classes=list(range(len(get_labels(args)))), y=all_label_ids_cpu)
        return dataset, cw
    
    return dataset


