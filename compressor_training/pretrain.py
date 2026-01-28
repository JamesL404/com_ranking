import random
import transformers
from peft import (
    LoraConfig,
)
from datasets import load_dataset, Dataset
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from training_utils import pretrain_tokenize_function, DataCollatorForDynamicPadding, train_model

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(data_args)
    
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}  # manually add this argument in the code
    if getattr(training_args, "report_to", None) in (None, [], ["none"], "none"):
        training_args.report_to = ["wandb"]
    if getattr(training_args, "logging_steps", 0) in (0, None):
        training_args.logging_steps = 20
    training_args.logging_first_step = True
    if hasattr(training_args, "dispatch_batches"):
        training_args.dispatch_batches = False
    if hasattr(training_args, "split_batches"):
        training_args.split_batches = False
    training_args.dataloader_num_workers = 0

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # check model_args.mem_size and min_tokens_for_lm
    assert (training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)) == 0, "training_args.fixed_mem_size must be a power of 2"    
    assert training_args.leave_tokens_for_lm <= training_args.min_tokens_for_lm, "leave_tokens_for_lm should be fewer than min_tokens_for_lm"

    
    memory_size = training_args.fixed_mem_size

    if data_args.data_path is None:
        raise ValueError("Please provide --data_path pointing to the training jsonl file.")

    train_file = data_args.data_path
    eval_file = data_args.eval_data_path or data_args.data_path

    print("Loading dataset...")

    dataset = load_dataset("json", data_files={"train": train_file}, streaming=True)
    train_dataset = dataset["train"]

    rng = random.Random(42)
    eval_buffer = []
    for row in train_dataset:
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        if len(eval_buffer) < 2000:
            eval_buffer.append({"text": text})
        else:
            j = rng.randint(0, len(eval_buffer))
            if j < 2000:
                eval_buffer[j] = {"text": text}
        if len(eval_buffer) == 2000:
            break
    eval_dataset = Dataset.from_list(eval_buffer)

    model = ICAE(model_args, training_args, lora_config)
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

    train_dataset = train_dataset.map(pretrain_tokenize_function, batched=True, batch_size=64, fn_kwargs={"model": model, "mem": MEM_TOKENS, "lm_ratio": training_args.lm_ratio})
    eval_dataset = eval_dataset.map(pretrain_tokenize_function, batched=True, fn_kwargs={"model": model, "mem": MEM_TOKENS})   # don't add lm in the dev set.

    data_collator = DataCollatorForDynamicPadding(
        model.pad_token_id,
        max_length=getattr(training_args, "model_max_length", None),
    )
    train_model(model, train_dataset, eval_dataset, training_args, data_collator)

main()
