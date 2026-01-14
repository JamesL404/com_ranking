import argparse
import json
import os
import random
import sys
import torch
from peft import LoraConfig
from safetensors.torch import load_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from compressor_training.modeling_icae_multi_span import ICAE, ModelArguments, TrainingArguments


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
    parser.add_argument("--compressor_base_model", required=True)
    parser.add_argument("--compressor_ckpt", required=True)
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
            tokenized = compressor.tokenizer(
                passage, truncation=True, max_length=4096, padding=False, return_attention_mask=False
            )
            input_ids = torch.tensor([tokenized["input_ids"]], device=device)
            memory_slots = compressor._compress(input_ids)

            prompt_ids = torch.tensor([[compressor.ae_token_id]], device=device)
            prompt_embs = compressor.tokens_to_embeddings(prompt_ids)
            output = torch.cat([memory_slots.to(prompt_embs), prompt_embs], dim=1)

            generate_ids = []
            past_key_values = None
            for _ in range(args.max_new_tokens):
                with compressor.icae.disable_adapter():
                    out = compressor.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                logit = out.logits[:, -1, : compressor.vocab_size - 1]
                past_key_values = out.past_key_values
                next_token_id = torch.argmax(logit, dim=-1)
                if next_token_id.item() == compressor.eos_id:
                    break
                output = compressor.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)
                generate_ids.append(next_token_id.item())

            reconstructed = compressor.tokenizer.decode(generate_ids)
            results.append({"passage": passage, "reconstruction": reconstructed})

    print(json.dumps({"results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
