# Flan-T5 on DialogSum: Full Fine-Tuning and LoRA (PEFT)

Reproducible fine-tuning of **Flan-T5** for dialogue summarization on **DialogSum** using the **Hugging Face Trainer**. The repo includes both **full fine-tuning** and **LoRA adapters** (PEFT) with identical preprocessing and evaluation (ROUGE), allowing apples-to-apples comparisons under fixed seeds.

## Highlights
- 🔁 Reproducible pipeline (`config.yaml`, fixed seeds, pinned deps)
- ⚙️ Two training modes: **full FT** and **LoRA** (parameter-efficient)
- 📈 Built-in **ROUGE** evaluation; identical prompts and decoding
- 🧱 Clean structure for hiring managers: clear scripts, docstrings, and CLI

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training

**Full fine-tuning**
```bash
python train.py --config config.yaml
# outputs to checkpoints/full/flant5-dialogsum-<timestamp>
```

**LoRA (PEFT)**
```bash
python train_lora.py --config config.yaml
# saves adapters to checkpoints/lora/flant5-dialogsum-lora-<timestamp>
```

## Inference

**Full FT checkpoint**
```bash
python inference.py --checkpoint checkpoints/full/flant5-dialogsum-<ts>   --text "A: ... B: ... A: ..." --max_new_tokens 200
```

**LoRA adapters on top of base**
```bash
python inference.py   --checkpoint google/flan-t5-small   --adapter checkpoints/lora/flant5-dialogsum-lora-<ts>   --text "A: ... B: ... A: ..." --max_new_tokens 200
```

## Evaluation
We report ROUGE on held-out DialogSum (same decode settings across models). To replicate, run:
- Train **raw baseline** = `google/flan-t5-small` (no FT)
- Train **full FT** via `train.py`
- Train **LoRA** via `train_lora.py`
- Evaluate with your preferred script/notebook or integrate a validation loop with `Trainer.predict`.

> Label padding is masked as `-100` to avoid loss on pad tokens; decoding uses the same prompt template for all runs.

## Notes
- Default LoRA targets are `["q","v"]` for portability across T5 variants. You can expand to `["q","k","v","o","wi_0","wi_1","wo"]` to also adapt FFN layers (larger adapter, often better).
- Mixed precision is enabled automatically: `bf16` on Ampere+, otherwise `fp16` if CUDA is present.

## Reference
- Dataset: [DialogSum (Hugging Face)](https://huggingface.co/datasets/knkarthick/dialogsum)
- Models: [Flan-T5 (Hugging Face)](https://huggingface.co/google)
- LoRA/PEFT: [PEFT library](https://github.com/huggingface/peft)
