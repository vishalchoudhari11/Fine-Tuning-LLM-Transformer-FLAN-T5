import os, time, torch, argparse, yaml
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
)
from utils import (
    set_seed, print_number_of_trainable_model_parameters,
    get_data_collator, build_tokenize_fn
)

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["full_ft"]["seed"])
    dataset = load_dataset(cfg["dataset_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    )
    model.config.use_cache = False  # silence warning during training

    print("✅ Dataset loaded:", dataset)
    print_number_of_trainable_model_parameters(model)

    # Tokenize
    tokenize_fn = build_tokenize_fn(
        tokenizer,
        cfg["max_source_length"],
        cfg["max_target_length"]
    )
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)

    # Collator
    data_collator = get_data_collator(tokenizer, model)

    # Output dir
    ts = str(int(time.time()))
    outdir = os.path.join(cfg["full_ft"]["output_root"], f"flant5-dialogsum-{ts}")
    os.makedirs(outdir, exist_ok=True)

    # Mixed precision flags
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    fp16 = torch.cuda.is_available() and not bf16

    args = TrainingArguments(
        output_dir=outdir,
        learning_rate=cfg["full_ft"]["learning_rate"],
        weight_decay=cfg["full_ft"]["weight_decay"],
        num_train_epochs=cfg["full_ft"]["num_train_epochs"],
        per_device_train_batch_size=cfg["full_ft"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["full_ft"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["full_ft"]["gradient_accumulation_steps"],
        evaluation_strategy=cfg["full_ft"]["eval_strategy"],
        save_strategy=cfg["full_ft"]["save_strategy"],
        logging_strategy=cfg["full_ft"]["log_strategy"],
        fp16=fp16,
        bf16=bf16,
        predict_with_generate=True,
        report_to=[],
        push_to_hub=False,
        seed=cfg["full_ft"]["seed"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    print(f"✅ Full fine-tuned model saved to: {outdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)