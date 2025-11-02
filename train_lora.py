import os, time, torch, argparse, yaml
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from utils import (
    set_seed, print_number_of_trainable_model_parameters,
    get_data_collator, build_tokenize_fn
)

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["lora"]["seed"])
    dataset = load_dataset(cfg["dataset_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    )
    base_model.config.use_cache = False

    # LoRA config (kept your defaults)
    lcfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        target_modules=cfg["lora"]["target_modules"],  # ["q","v"] by default
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    peft_model = get_peft_model(base_model, lcfg)
    print_number_of_trainable_model_parameters(peft_model)

    tokenize_fn = build_tokenize_fn(
        tokenizer,
        max_source_len=cfg["max_source_length"],
        max_target_len=cfg["max_target_length"]
    )
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
    data_collator = get_data_collator(tokenizer, peft_model)

    ts = str(int(time.time()))
    outdir = os.path.join(cfg["lora"]["output_root"], f"flant5-dialogsum-lora-{ts}")
    os.makedirs(outdir, exist_ok=True)

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    fp16 = torch.cuda.is_available() and not bf16

    args = TrainingArguments(
        output_dir=outdir,
        learning_rate=cfg["lora"]["learning_rate"],  # higher than full FT
        num_train_epochs=cfg["lora"]["num_train_epochs"],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        fp16=fp16,
        bf16=bf16,
        predict_with_generate=True,
        report_to=[],
        push_to_hub=False,
        seed=cfg["lora"]["seed"],
    )

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save only adapters
    peft_model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    print(f"✅ LoRA adapters saved to: {outdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)