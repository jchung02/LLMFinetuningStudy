import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix 
from datasets import load_metric
import pandas as pd
import pdb

from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    ElectraTokenizer,
    BertTokenizer,
    BertForTokenClassification,
    BertForSequenceClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification,
    ElectraForSequenceClassification
)
from tokenization_kobert import KoBertTokenizer

MODEL_CLASSES = {
    'kobert': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'kobert-tlink': (BertConfig, BertForSequenceClassification, KoBertTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertForTokenClassification, KoBertTokenizer),
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'kobert-lm': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'koelectra-base': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
    'koelectra-base-tlink': (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
    'koelectra-small': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
}
MODEL_PATH_MAP = {
    'kobert': 'monologg/kobert',
    'kobert-tlink': 'monologg/kobert',
    'distilkobert': 'monologg/distilkobert',
    'bert': 'bert-base-multilingual-cased',
    'kobert-lm': 'monologg/kobert-lm',
    'koelectra-base': 'monologg/koelectra-base-discriminator',
    'koelectra-base-tlink': 'monologg/koelectra-base-discriminator',
    'koelectra-small': 'monologg/koelectra-small-discriminator',
}

def get_test_texts(args, for_tlink=False):
    texts = []
    if args['do_eval']:
        the_arg = args['test_file']
    else:
        the_arg = args['val_file']
    with open(os.path.join(args["data_dir"], the_arg), 'r', encoding='utf-8') as f:
        for line in f:
            text, _ = line.split('\t')
            if for_tlink:
                text = text.split()
            else:
                text = list(text) 
            texts.append(text)
    return texts


def get_labels(args):
    ret = [label.strip() for label in open(os.path.join(args["data_dir"], args["label_file"]), 'r', encoding='utf-8')]
    return ret


def load_tokenizer(args):
    return MODEL_CLASSES[args["model_type"]][2].from_pretrained(args["model_name_or_path"])


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if not args["no_cuda"] and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args["seed"])


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec(labels, preds)

def compute_metrics_tlink(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec_tlink(labels, preds)

def f1_pre_rec_tlink(labels, preds):
    prec, rec, f1 = load_metric('precision'), load_metric('recall'), load_metric('f1')
    return {
        "(micro)precision": prec.compute(predictions=preds, references=labels, average='micro'),
        "(micro)recall": rec.compute(predictions=preds, references=labels, average='micro'),
        "(micro)f1": f1.compute(predictions=preds, references=labels, average='micro'),
        "(macro)precision": prec.compute(predictions=preds, references=labels, average='macro'),
        "(macro)recall": rec.compute(predictions=preds, references=labels, average='macro'),
        "(macro)f1": f1.compute(predictions=preds, references=labels, average='macro'),
        "(weighted)precision": prec.compute(predictions=preds, references=labels, average='weighted'),
        "(weighted)recall": rec.compute(predictions=preds, references=labels, average='weighted'),
        "(weighted)f1": f1.compute(predictions=preds, references=labels, average='weighted')
    }

def f1_pre_rec(labels, preds):
    return {
        "precision": precision_score(labels, preds, suffix=True),
        "recall": recall_score(labels, preds, suffix=True),
        "f1": f1_score(labels, preds, suffix=True)
    }


def show_report(labels, preds):
    return classification_report(labels, preds, suffix=True)

def show_report_tlink(labels, preds, class_names):
    ret = confusion_matrix(labels, preds, labels=class_names)
    return pd.DataFrame(ret, index=class_names, columns=class_names)
