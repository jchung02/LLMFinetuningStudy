import torch
import time
import sys

from data_loader import load_and_cache_examples
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from trainer import Trainer

import pdb

print_w_time = lambda elapsed: print("\t완료 ({}초 소요)".format(elapsed))

def train(args):
    print("> train_dataset 데이터 로딩: ", end="")
    start = time.time()
    args["data_dir"] = data_path + 'Train/AI모델링/'
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train", use_cache=False)
    print_w_time(time.time() - start)
    
    print("> dev_dataset 데이터 로딩: ", end="")
    start = time.time()
    args["data_dir"] = data_path + 'Validation/AI모델링/'
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev", use_cache=False)
    print_w_time(time.time() - start)
    
    print("> 학습객체 trainer 생성: ", end="")
    start = time.time()
    trainer = Trainer(args, train_dataset, dev_dataset, None)
    print_w_time(time.time() - start)

    print("> 학습(trainer.train)...")
    start = time.time()
    trainer.train()
    print_w_time(time.time() - start)

    print("> 학습된 모델 저장(trainer.save_model): ", end="")
    start = time.time()
    trainer.save_model()
    print_w_time(time.time() - start)


def test(args):
    print("> argument")
    print(args)
    print()

    print("> test_dataset 데이터 로딩: ", end="")
    start = time.time()
    args["data_dir"] = data_path + 'Test/AI모델링/'
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test", use_cache=False)
    print_w_time(time.time() - start)
    
    print("> 학습된 모델 불러오기(trainer.load_model): ", end="")
    start = time.time()
    trainer = Trainer(args, None, None, test_dataset)
    trainer.load_model()
    print_w_time(time.time() - start)
    
    print("> 테스트(trainer.evaluate)...")
    start = time.time()
    trainer.evaluate("test", "final", show_detail=True)
    print_w_time(time.time() - start)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:  $ python3 event.py train|test kobert|koelectra")
        exit()
    run_mode = sys.argv[1]
    model_type = sys.argv[2]
    if run_mode not in ["train", "test"]:
        print("Invalid run mode:", run_mode)
        exit()
    if model_type not in ["kobert", "koelectra"]:
        print("Invalid model type:", model_type)
        exit()

    if model_type == 'koelectra':
        model_type += '-base'

    data_path = './data_path/'
    args = {
        "task": "naver-ner",
        "model_dir": "./model_event_{}".format(model_type),
        "data_dir": data_path,
        "train_file": "event.train",
        "test_file": "event.test",
        "val_file": "event.val",
        "label_file": "label.event",
        "write_pred": True,
        "model_type": model_type,
        "seed": 42,
        "train_batch_size":64,
        "eval_batch_size": 64,
        "max_seq_len": 100,
        "learning_rate": 5e-5,
        "num_train_epochs": 40.0,
        "weight_decay": 0.0,
        "gradient_accumulation_steps": 1,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "patience": 2,
        "warmup_steps": 0,
        "logging_steps": 1000,
        "save_steps": 1000,
        "do_train": False,
        "do_eval": False,
        "no_cuda": False
    }
    args["model_name_or_path"] = MODEL_PATH_MAP[args["model_type"]]

    print("> 토크나이저 로딩: ", end="")
    start = time.time()
    tokenizer = load_tokenizer(args)
    print_w_time(time.time() - start)

    if run_mode == 'train':
        args["do_train"] = True
        args["pred_dir"] = "./validation_event_{}".format(model_type)
        train(args)
    elif run_mode == 'test':
        args["do_eval"] = True
        args["pred_dir"] = "./test_event_{}".format(model_type)
        test(args)


