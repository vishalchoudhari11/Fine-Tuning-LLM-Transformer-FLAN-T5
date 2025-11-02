import random, os
import numpy as np
import torch
import evaluate
from transformers import DataCollatorForSeq2Seq

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_number_of_trainable_model_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0
    print(f"Trainable params: {trainable:,}")
    print(f"Total params: {total:,}")
    print(f"Percentage trainable: {pct:.2f}%")

def get_data_collator(tokenizer, model):
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

def build_tokenize_fn(tokenizer, max_source_len=512, max_target_len=128):
    prefix = "Summarize the following conversation.\n\n"
    suffix = "\n\nSummary: "
    pad_token_id = tokenizer.pad_token_id

    def _to_ids(batch):
        inputs = [prefix + d + suffix for d in batch["dialogue"]]
        model_inputs = tokenizer(
            inputs, max_length=max_source_len, truncation=True, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["summary"], max_length=max_target_len, truncation=True, padding="max_length"
            )
        # mask pad tokens in labels to -100
        labels_ids = []
        for row in labels["input_ids"]:
            labels_ids.append([(lid if lid != pad_token_id else -100) for lid in row])
        model_inputs["labels"] = labels_ids
        return model_inputs

    return _to_ids

def compute_rouge():
    rouge = evaluate.load("rouge")
    def _compute(eval_pred):
        preds, labels = eval_pred
        # Replace -100 back to pad_token_id for decoding safety if needed upstream
        return rouge.compute(predictions=preds, references=labels, use_stemmer=True)
    return _compute