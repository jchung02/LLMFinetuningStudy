import os
import shutil
import logging
from tqdm import tqdm, trange
import pdb

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import compute_metrics_tlink, get_labels, get_test_texts, show_report_tlink, MODEL_CLASSES

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, tokenizer=None, class_weights=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_lst = get_labels(args)
        self.num_labels = len(self.label_lst)
        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args["model_type"]]

        self.config = self.config_class.from_pretrained(args["model_name_or_path"],
                                                        num_labels=self.num_labels,
                                                        finetuning_task=args["task"],
                                                        id2label={str(i): label for i, label in enumerate(self.label_lst)},
                                                        label2id={label: i for i, label in enumerate(self.label_lst)})
        self.model = self.model_class.from_pretrained(args["model_name_or_path"], config=self.config)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu"
       
        # class weights
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = torch.Tensor(self.class_weights)
            self.class_weights = self.class_weights.to(self.device)

        if tokenizer:
            self.model.resize_token_embeddings(len(tokenizer))
        
        self.model.to(self.device)

        self.test_texts = None
        if args["write_pred"]:
            self.test_texts = get_test_texts(args, for_tlink=True)
            if os.path.exists(args["pred_dir"]):
                shutil.rmtree(args["pred_dir"])

    def _compute_logits_loss(self, batch):
        batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[3]}
        if self.args["model_type"] != 'distilkobert':
            inputs['token_type_ids'] = batch[2]
        outputs = self.model(**inputs)
        loss, logits = outputs[0], outputs[1]
        labels = batch[3]
        return logits, loss, labels

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args["train_batch_size"])

        if self.args["max_steps"] > 0:
            t_total = self.args["max_steps"]
            self.args["num_train_epochs"] = self.args["max_steps"] // (len(train_dataloader) // self.args["gradient_accumulation_steps"]) + 1
        else:
            t_total = len(train_dataloader) // self.args["gradient_accumulation_steps"] * self.args["num_train_epochs"]

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args["weight_decay"]},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args["learning_rate"], eps=self.args["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args["warmup_steps"], num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args["num_train_epochs"])
        logger.info("  Total train batch size = %d", self.args["train_batch_size"])
        logger.info("  Gradient Accumulation steps = %d", self.args["gradient_accumulation_steps"])
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args["logging_steps"])
        logger.info("  Patience = %d", self.args["patience"])
        logger.info("  Save steps = %d", self.args["save_steps"])

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args["num_train_epochs"]), desc="Epoch")

        to_stop = False
        trigger_times = 0
        last_loss = None
        patience = self.args['patience']
        for ei, _ in enumerate(train_iterator):
            print("[Epoch] {}/{}".format(ei+1, self.args['num_train_epochs']))
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                logits, loss, labels = self._compute_logits_loss(batch)

                if self.args["gradient_accumulation_steps"] > 1:
                    loss = loss / self.args["gradient_accumulation_steps"]

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args["max_grad_norm"])

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args["logging_steps"] > 0 and global_step % self.args["logging_steps"] == 0:
                        eval_results = self.evaluate("dev", global_step)
                        eval_loss = eval_results['loss']
                        if last_loss == None or eval_loss > last_loss:
                            trigger_times += 1
                        last_loss = eval_loss
                        print("model checked with dev dataset (eval loss: {}, #trigger: {}/{})".format(eval_loss, trigger_times, patience))
                        if patience >0 and trigger_times >= patience:
                            print("Early stopped!")
                            to_stop = True

                    if self.args["save_steps"] > 0 and global_step % self.args["save_steps"] == 0:
                        self.save_model()
                        print("model saved.")

                if to_stop:
                    break

                if 0 < self.args["max_steps"] < global_step:
                    epoch_iterator.close()
                    break

            if to_stop or 0 < self.args["max_steps"] < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode, step, show_detail=False):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args["eval_batch_size"])

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args["eval_batch_size"])
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                logits, loss, labels = self._compute_logits_loss(batch)
                eval_loss += loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        preds = np.argmax(preds, axis=1)
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = []
        preds_list = []

        for i in range(out_label_ids.shape[0]):
            out_label_list.append(slot_label_map[out_label_ids[i]])
            preds_list.append(slot_label_map[preds[i]])

        if self.args["write_pred"]:
            if not os.path.exists(self.args["pred_dir"]):
                os.mkdir(self.args["pred_dir"])

            with open(os.path.join(self.args["pred_dir"], "pred_{}.txt".format(step)), "w", encoding="utf-8") as f:
                for text, true_label, pred_label in zip(self.test_texts, out_label_list, preds_list):
                    f.write("{} {} {}\n".format(text, true_label, pred_label))

        result = compute_metrics_tlink(out_label_ids, preds)
        results.update(result)

        logger.info("***** Eval results *****")
        print("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            print("\t{} = {}".format(key, str(results[key])))
        if show_detail:
            print(show_report_tlink(out_label_list, preds_list, self.label_lst))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args["model_dir"]):
            os.makedirs(self.args["model_dir"])
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args["model_dir"])

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args["model_dir"], 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args["model_dir"])

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args["model_dir"]):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args["model_dir"])
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

