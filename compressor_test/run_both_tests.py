import argparse
import json
import os
import subprocess
import sys
from tqdm import tqdm
from huggingface_hub import snapshot_download


def run_cmd(cmd):
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressor_base_model", required=True)
    parser.add_argument("--compressor_ckpt", required=True)
    parser.add_argument("--instruct_model", required=True)
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--sample_count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed_mem_size", type=int, default=8)
    parser.add_argument("--mean_compression_rate", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    instruct_local = snapshot_download(
        args.instruct_model,
        local_dir=args.hf_cache_dir,
    )

    base_cmd = [
        sys.executable,
        "reconstruct_from_compressor.py",
        "--compressor_base_model",
        args.compressor_base_model,
        "--compressor_ckpt",
        args.compressor_ckpt,
        "--instruct_model",
        instruct_local,
        "--data_path",
        args.data_path,
        "--sample_count",
        str(args.sample_count),
        "--seed",
        str(args.seed),
        "--fixed_mem_size",
        str(args.fixed_mem_size),
        "--mean_compression_rate",
        str(args.mean_compression_rate),
        "--max_new_tokens",
        str(args.max_new_tokens),
    ]
    icae_cmd = [
        sys.executable,
        "reconstruct_with_icae.py",
        "--compressor_base_model",
        args.compressor_base_model,
        "--compressor_ckpt",
        args.compressor_ckpt,
        "--data_path",
        args.data_path,
        "--sample_count",
        str(args.sample_count),
        "--seed",
        str(args.seed),
        "--fixed_mem_size",
        str(args.fixed_mem_size),
        "--mean_compression_rate",
        str(args.mean_compression_rate),
        "--max_new_tokens",
        str(args.max_new_tokens),
    ]

    steps = [
        ("from_compressor", base_cmd),
        ("with_icae", icae_cmd),
    ]
    results = {}
    for name, cmd in tqdm(steps, desc="Running tests"):
        results[name] = run_cmd(cmd)

    combined = {"from_compressor": results["from_compressor"], "with_icae": results["with_icae"]}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
