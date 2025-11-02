import argparse, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

def load_model(checkpoint_dir, adapter_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto"
    )
    if adapter_dir:
        base = PeftModel.from_pretrained(base, adapter_dir)
    base.eval()
    return base, tokenizer

def summarize(model, tokenizer, dialogue, max_new_tokens=200, device=None):
    prompt = f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device or "cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        out = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to full FT checkpoint (or base for LoRA)")
    p.add_argument("--adapter", default=None, help="Optional LoRA adapter directory")
    p.add_argument("--text", required=True, help="Raw dialogue text to summarize")
    p.add_argument("--max_new_tokens", type=int, default=200)
    args = p.parse_args()

    model, tok = load_model(args.checkpoint, args.adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(summarize(model, tok, args.text, max_new_tokens=args.max_new_tokens, device=device))