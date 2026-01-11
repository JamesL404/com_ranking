import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import torch
import torch.nn as nn
import random
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    get_peft_model,
)
from torch.nn.functional import gelu
import math
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-v0.1")
    lora_r: int = field(
        default=128,
        metadata={"help": "lora rank"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "lora dropout"}
    )
    train: bool = field(
        default=True,
        metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset to quickly verify the training process"})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Optional path to the eval data. Defaults to train data if not set."})
    max_eval_samples: Optional[int] = field(default=2000, metadata={"help": "Max eval samples for streaming datasets."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    fixed_mem_size: int = field(
        default=128,
        metadata={"help": "Enalbing the fixed mem size."},
    )
    mean_compression_rate: int = field(
        default=4,
        metadata={"help": "Mean compression rate; default=4"},
    )
    min_tokens_for_lm: int = field(
        default=64,
        metadata={"help": "Minimum tokens for lm objective learning"},
    )
    leave_tokens_for_lm: int = field(
        default=8,
        metadata={"help": "Leave some tokens without loss for lm objective"},
    )
    lm_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio for LM training."},
    )
    add_special_token_for_lm: bool = field(
        default=False,
        metadata={"help": "Add a special token for the prompt of language modeling; default: False"},
    )
    restore_from: str = field(
        default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"}
    )

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class ICAE(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        attn_impl = {"attn_implementation": "sdpa"}
        self.icae = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16,
            resume_download=True,
            trust_remote_code=True,
            **attn_impl,
        )
        
        self.training = self.model_args.train    
        
        if self.training:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16,
                resume_download=True,
                trust_remote_code=True,
                **attn_impl,
            )

        self.vocab_size = self.icae.config.vocab_size + 1
        self.pad_token_id = self.vocab_size - 1
        self.mean_compression_rate = training_args.mean_compression_rate

        self.mem_size = self.training_args.fixed_mem_size
        self.vocab_size_with_mem = self.vocab_size + self.mem_size

        self.ae_token_id = self.vocab_size_with_mem + 0
        self.lm_token_id = self.vocab_size_with_mem + 1
        self.ft_token_id = self.vocab_size_with_mem + 2        

        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 3) 
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True)
        self.bos_id = self.icae.config.bos_token_id if self.icae.config.bos_token_id is not None else self.tokenizer.bos_token_id
        self.eos_id = self.icae.config.eos_token_id if self.icae.config.eos_token_id is not None else self.tokenizer.eos_token_id
        
        self.dim = self.icae.config.hidden_size
        self.icae = get_peft_model(self.icae, lora_config)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory_token_embed = nn.Embedding(self.mem_size + 3, self.dim, padding_idx=None)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.append_sequence = torch.arange(self.vocab_size, self.vocab_size + self.mem_size, dtype=torch.long, device=device).unsqueeze(0)
        
        if self.training_args.save_safetensors:
            self._untie_shared_embeddings()
        
        if self.training:
            self.init()


    def init(self):
        print("Freezing the decoder...")
        freeze_model(self.decoder)
        self.decoder.eval()
        print_trainable_parameters(self)
        if self.training_args.restore_from is not None and self.training_args.restore_from != "":
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")
        print("Enabling gradient checkpointing...")
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def _untie_shared_embeddings(self):
        def _maybe_untie(module):
            if not hasattr(module, "get_input_embeddings") or not hasattr(module, "get_output_embeddings"):
                return
            embed = module.get_input_embeddings()
            lm_head = module.get_output_embeddings()
            if embed is None or lm_head is None:
                return
            if embed.weight.data_ptr() == lm_head.weight.data_ptr():
                lm_head.weight = nn.Parameter(embed.weight.detach().clone())
                if hasattr(module, "config"):
                    module.config.tie_word_embeddings = False

        _maybe_untie(self.icae)
        if self.training:
            _maybe_untie(self.decoder)
                
        
    def compute_num_segments(self, total_length):
        assert total_length > 0
        num_segments = math.ceil(total_length / (self.mem_size * self.mean_compression_rate))
        return num_segments


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        prompt_answer_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)
        
        prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer_ids)
        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((batch_size, max_compressed_length, self.dim)).to(prompt_answer_embs)
        
        for segment_idx in range(num_segments):
            
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            mem_append = self.append_sequence.expand(segment_input_ids.size(0), -1)
            segment_input_ids = torch.cat([segment_input_ids, mem_append], dim=1)
            mem_flag = segment_input_ids >= self.vocab_size

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding)

            segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]

            mem_outputs = segment_compress_outputs[:, -self.mem_size:, :]
            start = segment_idx * self.mem_size
            end = self.mem_size * (segment_idx + 1)
            compress_outputs[:, start:end, :] = mem_outputs
            
            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()
            
        decoder_mem_flag = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size + self.mem_size)

        for b in range(batch_size):
            prompt_answer_embs[b, decoder_mem_flag[b]] = compress_outputs[b]
        special_prompt = prompt_answer_ids >= self.vocab_size_with_mem
        prompt_answer_embs[special_prompt] = self.memory_token_embed(prompt_answer_ids[special_prompt] - self.vocab_size).to(prompt_answer_embs)
        
        if self.training:
            decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
        else:
            with self.icae.disable_adapter():
                decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, output_hidden_states=True)


        logits = decoder_outputs.logits
        effective_logits = logits[:,:-1,:].reshape(-1, logits.size(-1))
        target_ids = labels[:,1:].reshape(-1)
        loss = self.loss_fct(effective_logits, target_ids)
        return {"loss": loss, "logits": logits}
    
    
    def tokens_to_embeddings(self, token_ids):
        embeddings = self.icae.get_base_model().model.embed_tokens(token_ids)
        special_flags = token_ids >= self.vocab_size
        embeddings[special_flags] = self.memory_token_embed(token_ids[special_flags] - self.vocab_size).to(embeddings)
        return embeddings
        
    
    def _compress(
        self,
        input_ids: torch.LongTensor = None
    ):

        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)
        
        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((batch_size, max_compressed_length, self.dim))
        
        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            mem_append = self.append_sequence.expand(segment_input_ids.size(0), -1)
            segment_input_ids = torch.cat([segment_input_ids, mem_append], dim=1)
            mem_flag = segment_input_ids >= self.vocab_size

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding)

            segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]

            mem_outputs = segment_compress_outputs[:, -self.mem_size:, :]
            start = segment_idx * self.mem_size
            end = self.mem_size * (segment_idx + 1)
            compress_outputs[:, start:end, :] = mem_outputs
            
            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()
        
        return compress_outputs
