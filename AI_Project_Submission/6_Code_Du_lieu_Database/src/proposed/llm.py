import torch
from typing import List, Dict
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from benchmark import (
    SYS_MEDQA_RAG,
    SYS_MEDQA_NO_RAG,
    SYS_PUBMED_BUILTIN,
    MAX_CONTEXT_TOKENS,
    MAX_NEW_TOKENS,
    format_context,
    _truncate_to_tokens,
)
from unsloth import FastLanguageModel


class BaseGenerator:
    def __init__(
        self,
        model_path: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        lora_path: str = None,
        is_unsloth: bool = True,
    ):
        self.model_path = model_path
        self.lora_path = lora_path
        print(f"[+] Loading LLM: {model_path} with LoRA: {lora_path}")
        
        if is_unsloth:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=MAX_CONTEXT_TOKENS,
                dtype=torch.float16,
                load_in_4bit=True,
            )
            if lora_path:
                self.model.load_adapter(lora_path)
            FastLanguageModel.for_inference(self.model)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto"
            )

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def generate(
        self,
        context_docs: List[Dict],
        question: str,
        options: dict = None,
        dataset_type: str = "medqa",
    ) -> tuple[str, float]:
        import time
        # Build context string
        context_str = format_context(context_docs) if context_docs else ""

        # Build prompt based on dataset
        if dataset_type == "medqa":
            options_text = "\n".join([f"{k}) {v}" for k, v in (options or {}).items()])
            user_question = f"Question:\n{question}\n\nOptions:\n{options_text}"

            use_rag = bool(context_docs)
            sys_prompt = SYS_MEDQA_RAG if use_rag else SYS_MEDQA_NO_RAG

            if use_rag and context_str:
                sys_len = len(self.tokenizer.encode(sys_prompt))
                q_len = len(self.tokenizer.encode(user_question))
                budget = MAX_CONTEXT_TOKENS - sys_len - q_len - 50
                truncated_ctx = _truncate_to_tokens(
                    context_str, self.tokenizer, max(budget, 0)
                )
                user_content = f"Context:\n{truncated_ctx}\n\n{user_question}"
            else:
                user_content = user_question

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ]
        elif dataset_type == "pubmedqa":
            sys_len = len(self.tokenizer.encode(SYS_PUBMED_BUILTIN))
            q_len = len(self.tokenizer.encode(question))
            budget = MAX_CONTEXT_TOKENS - sys_len - q_len - 50
            truncated_ctx = _truncate_to_tokens(
                context_str, self.tokenizer, max(budget, 0)
            )

            user_content = f"Context:\n{truncated_ctx}\n\nQuestion: {question}\nAnswer:"
            messages = [
                {"role": "system", "content": SYS_PUBMED_BUILTIN},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [{"role": "user", "content": question}]

        # Standardize formatting via chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without native chat template support
            prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\nAssistant: "

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.05,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        end_time = time.perf_counter()

        new_tokens = outputs[0, input_len:]
        generated_tokens_count = max(len(new_tokens), 1)
        latency_s = end_time - start_time
        throughput = generated_tokens_count / latency_s

        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return decoded.strip(), throughput

