import argparse
import json
import os
import random
import sys
import torch
from peft import LoraConfig
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from compressor_training.modeling_icae_multi_span import ICAE, ModelArguments, TrainingArguments


def build_prompt(tokenizer, instruction):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    return tokenizer(instruction, return_tensors="pt").input_ids


def iter_passages(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                except json.JSONDecodeError:
                    text = ""
            else:
                text = line
            text = text.strip()
            if text:
                yield text


def sample_passages(data_path, count, seed):
    rng = random.Random(seed)
    sample = []
    for i, text in enumerate(iter_passages(data_path), start=1):
        if len(sample) < count:
            sample.append(text)
            continue
        j = rng.randint(1, i)
        if j <= count:
            sample[j - 1] = text
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressor_base_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--compressor_ckpt", required=True)
    parser.add_argument("--instruct_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--passage", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--sample_count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--fixed_mem_size", type=int, default=8)
    parser.add_argument("--mean_compression_rate", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_args = ModelArguments(model_name_or_path=args.compressor_base_model, train=False)
    training_args = TrainingArguments(
        output_dir="./tmp",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        fixed_mem_size=args.fixed_mem_size,
        mean_compression_rate=args.mean_compression_rate,
    )
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    compressor = ICAE(model_args, training_args, lora_config)
    state_dict = load_file(args.compressor_ckpt)
    compressor.load_state_dict(state_dict, strict=False)
    compressor = compressor.to(device)
    compressor.eval()

    instr_tokenizer = AutoTokenizer.from_pretrained(args.instruct_model, trust_remote_code=True)
    instr_model = AutoModelForCausalLM.from_pretrained(
        args.instruct_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    instr_model.generation_config.temperature = None
    instr_model.generation_config.top_p = None
    instr_model.generation_config.top_k = None
    instr_model.eval()

    proj = None
    if compressor.icae.config.hidden_size != instr_model.config.hidden_size:
        proj = torch.nn.Linear(
            compressor.icae.config.hidden_size,
            instr_model.config.hidden_size,
            bias=False,
        ).to(device=device, dtype=instr_model.dtype)

    passages = []
    if args.passage:
        passages = [args.passage]
    elif args.data_path:
        passages = sample_passages(args.data_path, args.sample_count, args.seed)
    else:
        raise ValueError("Provide --passage or --data_path.")

    results = []
    with torch.no_grad():
        for passage in passages:
            tokenized = compressor.tokenizer(passage, truncation=True, max_length=4096, padding=False, return_attention_mask=False)
            input_ids = torch.tensor([tokenized["input_ids"]], device=device)
            memory_slots = compressor._compress(input_ids)

            instruction = "Reconstruct the original passage from the compressed memory tokens."
            prompt_ids = build_prompt(instr_tokenizer, instruction).to(device)
            prompt_embs = instr_model.get_input_embeddings()(prompt_ids)

            mem_embs = memory_slots.to(prompt_embs)
            if proj is not None:
                mem_embs = proj(mem_embs)
            inputs_embeds = torch.cat([mem_embs, prompt_embs], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)

            output_ids = instr_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=instr_tokenizer.eos_token_id,
            )

            generated = instr_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            results.append({"passage": passage, "reconstruction": generated})

    print(json.dumps({"results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
