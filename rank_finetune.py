import transformers
import torch
import re
from tqdm import tqdm
from peft import LoraConfig
from datasets import load_dataset, Dataset
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from training_utils import instruct_ft_tokenize_function, DataCollatorForDynamicPadding, train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compress_passage(model, tokenizer, text: str):
    """
    Compress one passage into ICAE latent memory slots using model._compress().
    Returns latent tensor of shape [mem_slots, hidden_dim].
    """
    if not text.strip():
        return None
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False
        ).to(device)
        latent = model._compress(tokens["input_ids"])
    return latent.cpu()


def replace_with_latents(prompt: str, n_latents: int):
    """
    Replace each [i] <PASSAGE> placeholder with [i] <LATENT_i>.
    """
    new_prompt = prompt
    for i in range(1, n_latents + 1):
        new_prompt = re.sub(rf"\[{i}\]\s*<PASSAGE>", f"[{i}] <LATENT_{i}>", new_prompt)
    return new_prompt.strip()


def preprocess_dataset(dataset, model, tokenizer):
    """
    Compress each passage and replace placeholders with <LATENT_i>.
    Produces a new dataset for ICAE instruction fine-tuning.
    """
    processed = []
    for ex in tqdm(dataset, desc="Compressing passages"):
        passages = ex.get("Input", [])
        prompt = ex.get("Prompt", "")
        answer = ex.get("Answer", "")

        # Skip corrupted entries
        if not passages or not prompt or not answer:
            continue

        latents = []
        for p in passages:
            latent = compress_passage(model, tokenizer, p)
            if latent is not None:
                latents.append(latent)

        if not latents:
            continue

        prompt_with_latents = replace_with_latents(prompt, len(latents))
        processed.append({
            "input": "<DUMMY>",  # ICAE expects this field
            "prompt": prompt_with_latents,
            "answer": answer,
            "latent_count": len(latents)
        })

    return Dataset.from_list(processed)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("\n=== Model / Data / Training Args ===")
    print(model_args)
    print(data_args)

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Ensure mem_size is power of 2
    assert (training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)) == 0, \
        "training_args.fixed_mem_size must be a power of 2"

    memory_size = training_args.fixed_mem_size

    # ===== Load and split dataset =====
    print("\nLoading RankZephyr dataset...")
    raw_dataset = load_dataset("json", data_files="/home/haowei/icae/rankzephyr_processed.jsonl")["train"]
    raw_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    raw_train, raw_eval = raw_dataset["train"], raw_dataset["test"]

    # ===== Initialize ICAE =====
    print("\nInitializing ICAE model and tokenizer...")
    model = ICAE(model_args, training_args, lora_config).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

    # ===== Compress passages =====
    print("\nCompressing passages with ICAE encoder (may take a while)...")
    train_dataset = preprocess_dataset(raw_train, model, tokenizer)
    eval_dataset = preprocess_dataset(raw_eval, model, tokenizer)

    print(f"✅ Compression done. {len(train_dataset)} train / {len(eval_dataset)} eval samples.")

    # ===== Tokenization for ICAE instruction FT =====
    print("\nTokenizing datasets for instruction fine-tuning...")
    train_dataset = train_dataset.map(
        instruct_ft_tokenize_function,
        batched=True,
        fn_kwargs={"model": model, "mem": MEM_TOKENS}
    )
    eval_dataset = eval_dataset.map(
        instruct_ft_tokenize_function,
        batched=True,
        fn_kwargs={"model": model, "mem": MEM_TOKENS}
    )

    # ===== Train =====
    print("\nStarting ICAE fine-tuning with RankZephyr data...")
    data_collator = DataCollatorForDynamicPadding(model.pad_token_id)
    train_model(model, train_dataset, eval_dataset, training_args, data_collator=data_collator)


if __name__ == "__main__":
    main()
