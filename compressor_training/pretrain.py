import transformers
from peft import (
    LoraConfig,
)
from datasets import load_dataset
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from training_utils import pretrain_tokenize_function, DataCollatorForDynamicPadding, train_model

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(data_args)

    if hasattr(training_args, "dispatch_batches"):
        training_args.dispatch_batches = False
        print("dispatch_batches set to False for streaming dataset.")
    if hasattr(training_args, "split_batches"):
        training_args.split_batches = False
        print("split_batches set to False to avoid cross-rank batch splitting.")
    training_args.dataloader_num_workers = 0
        
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    if getattr(training_args, "report_to", None) in (None, [], ["none"], "none"):
        training_args.report_to = ["wandb"]
        print("report_to set to ['wandb'] for logging.")
    training_args.logging_strategy = "steps"
    if getattr(training_args, "logging_steps", 0) in (0, None):
        training_args.logging_steps = 20
    training_args.logging_first_step = True

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    assert (training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)) == 0, "training_args.fixed_mem_size must be a power of 2"    
    assert training_args.leave_tokens_for_lm <= training_args.min_tokens_for_lm, "leave_tokens_for_lm should be fewer than min_tokens_for_lm"

    memory_size = training_args.fixed_mem_size

    if data_args.data_path is None:
        raise ValueError("Please provide --data_path pointing to the training jsonl file.")

    train_file = data_args.data_path
    eval_file = data_args.eval_data_path or data_args.data_path

    print("Loading dataset...")

    dataset = load_dataset("json", data_files={"train": train_file, "eval": eval_file}, streaming=True)
    train_dataset = dataset["train"].filter(lambda x: x.get("text") is not None and len(x["text"].strip()) > 0)
    eval_dataset = dataset["eval"].filter(lambda x: x.get("text") is not None and len(x["text"].strip()) > 0)
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.take(data_args.max_eval_samples)

    if training_args.max_steps == -1:
        training_args.max_steps = 20000
        print(f"max_steps not provided; defaulting to {training_args.max_steps} for streaming dataset.")

    model = ICAE(model_args, training_args, lora_config)
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

    train_dataset = train_dataset.map(pretrain_tokenize_function, batched=True, batch_size=64, fn_kwargs={"model": model, "mem": MEM_TOKENS, "lm_ratio": training_args.lm_ratio})
    eval_dataset = eval_dataset.map(pretrain_tokenize_function, batched=True, fn_kwargs={"model": model, "mem": MEM_TOKENS})

    data_collator = DataCollatorForDynamicPadding(
        model.pad_token_id, 
        max_length=getattr(training_args, "model_max_length", 5120)
    )
    
    train_model(model, train_dataset, eval_dataset, training_args, data_collator)

main()
