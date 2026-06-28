import os
import re
import shutil
import json
import numpy as np
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint
from unsloth.chat_templates import train_on_responses_only
from transformers import (
    TrainerCallback,
    EarlyStoppingCallback,
    TrainerControl,
    TrainerState,
)

# ==========================================
# 1. CONFIGURATION & KAGGLE PATHS
# ==========================================
max_seq_length = 1536  # Maximum sequence length for training
dtype = None  # Auto-detect precision (fp16/bf16)
load_in_4bit = True  # Enable 4-bit quantization (QLoRA)

OUTPUT_DIR = "/kaggle/working/models/qwen_chatdoctor_lora_new_dataset"
MERGED_DIR = "/kaggle/working/models/qwen_chatdoctor_merged"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

# ==========================================
# 2. LOAD MODEL & TOKENIZER
# ==========================================
print("Loading model and tokenizer...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ==========================================
# 3. PREPARE DATASET
# ==========================================
print("Loading PubMedQA dataset for RAG-SFT...")

dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

EOS_TOKEN = tokenizer.eos_token


def filter_by_response_length(example):
    c_texts = example["context"]["contexts"]
    context_str = "\n\n".join(c_texts)

    question_text = f"Context:\n{context_str}\n\nQuestion:\n{example['question']}"
    response_text = f"{example['long_answer']}\n\nConclusion: {example['final_decision'].capitalize()}."

    tokenized_resp = tokenizer(
        str(response_text), truncation=False, add_special_tokens=False
    )
    tokenized_quest = tokenizer(
        str(question_text), truncation=False, add_special_tokens=False
    )

    total_length = len(tokenized_resp["input_ids"]) + len(tokenized_quest["input_ids"])
    return total_length < max_seq_length - 10 - 25


print("Filtering dataset...")
dataset = dataset.filter(filter_by_response_length, num_proc=2)


def formatting_prompts_func(examples):
    questions = examples["question"]
    contexts_list = examples["context"]
    long_answers = examples["long_answer"]
    final_decisions = examples["final_decision"]

    texts = []

    for q, ctx, la, fd in zip(questions, contexts_list, long_answers, final_decisions):
        c_texts = ctx["contexts"]
        context_str = "\n\n".join(c_texts)

        input_text = f"Context:\n{context_str}\n\nQuestion:\n{q}"
        output_text = f"{la}\n\nConclusion: {fd.capitalize()}."

        messages = [
            {
                "role": "system",
                "content": "You are an expert medical AI assistant. You will be provided with medical abstracts as context. Your task is to carefully read the context and use it to answer the user's question. Formulate a detailed explanation based on the context, and conclude with a final decision of Yes, No, or Maybe.",
            },
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return {"text": texts}


print("Formatting dataset...")
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=5,
)

# Keep raw eval examples BEFORE chat-template formatting for generation eval
print("Loading raw eval split for generation-based evaluation...")
raw_dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
raw_dataset = raw_dataset.filter(filter_by_response_length, num_proc=2)

print("Splitting dataset into train and eval...")
split_dataset = dataset.train_test_split(test_size=200, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Get the same 200 raw examples (same seed, same split)
raw_split = raw_dataset.train_test_split(test_size=200, seed=42)
raw_eval = raw_split["test"]


# ==========================================
# 4. HELPER: EXTRACT ANSWER
# ==========================================
def extract_answer(text: str):
    """
    Robust parser to extract final decision (Yes, No, Maybe) for PubMedQA.
    """
    match = re.search(r"Conclusion:\s*(Yes|No|Maybe)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Fallback to the very end of the generated text
    match_end = re.search(
        r"\b(yes|no|maybe)\b[\.\s]*(?:<\|im_end\|>)?$", text, re.IGNORECASE
    )
    if match_end:
        return match_end.group(1).lower()

    return None


# ==========================================
# 5. GENERATION-BASED EVAL CALLBACK
# ==========================================
class GenerationEvalCallback(TrainerCallback):
    """
    At each eval step, run model.generate() on the raw eval set and
    compute accuracy from the 'Conclusion: Yes/No/Maybe' pattern.
    Injects the result into trainer.state so EarlyStoppingCallback
    and checkpoint logic can consume it.
    """

    def __init__(
        self, raw_eval_dataset, tokenizer, max_new_tokens=256, gen_batch_size=4
    ):
        self.raw_eval = raw_eval_dataset
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.gen_batch_size = gen_batch_size

    def _build_prompt(self, example):
        """Build the prompt-only (no assistant answer) for generation."""
        c_texts = example["context"]["contexts"]
        context_str = "\n\n".join(c_texts)
        input_text = f"Context:\n{context_str}\n\nQuestion:\n{example['question']}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert medical AI assistant. You will be provided with "
                    "medical abstracts as context. Your task is to carefully read the context "
                    "and use it to answer the user's question. Formulate a detailed explanation "
                    "based on the context, and conclude with a final decision of Yes, No, or Maybe."
                ),
            },
            {"role": "user", "content": input_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # adds <|im_start|>assistant\n
        )
        return prompt

    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs
    ):
        print("\n[GenerationEvalCallback] Running generation-based evaluation...")

        # Switch to inference mode (Unsloth optimisation)
        FastLanguageModel.for_inference(model)
        model.eval()

        device = next(model.parameters()).device
        tokenizer = self.tokenizer

        prompts = [self._build_prompt(ex) for ex in self.raw_eval]
        labels = [ex["final_decision"].strip().lower() for ex in self.raw_eval]

        correct = 0
        total = len(prompts)

        for i in range(0, total, self.gen_batch_size):
            batch_prompts = prompts[i : i + self.gen_batch_size]
            batch_labels = labels[i : i + self.gen_batch_size]

            encodings = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length - self.max_new_tokens,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **encodings,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # greedy decoding for reproducibility
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            input_len = encodings["input_ids"].shape[1]
            generated = outputs[:, input_len:]
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for pred_text, true_label in zip(decoded, batch_labels):
                pred_ans = extract_answer(pred_text)
                if pred_ans == true_label:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        print(
            f"[GenerationEvalCallback] Step {state.global_step} | "
            f"Accuracy: {accuracy:.4f} ({correct}/{total})"
        )

        # Inject into trainer logs so EarlyStoppingCallback sees it
        if state.log_history:
            state.log_history[-1]["eval_accuracy"] = accuracy
        else:
            state.log_history.append(
                {"eval_accuracy": accuracy, "step": state.global_step}
            )

        # Also inject into the last metrics dict the trainer checks
        kwargs.get("metrics", {})["eval_accuracy"] = accuracy

        # Switch back to training mode
        FastLanguageModel.for_training(model)
        model.train()


# ==========================================
# 6. CHECKPOINT CALLBACK (LAST + BEST)
# ==========================================
class BestCheckpointCallback(TrainerCallback):
    """
    After each evaluation:
    - Always copies the latest checkpoint → checkpoint-last
    - If accuracy improved → copies to checkpoint-best and saves metadata
    """

    LAST_DIR = os.path.join(OUTPUT_DIR, "checkpoint-last")
    BEST_DIR = os.path.join(OUTPUT_DIR, "checkpoint-best")
    META_FILE = os.path.join(OUTPUT_DIR, "best_checkpoint_meta.json")

    def __init__(self):
        self.best_accuracy = -1.0

    def _get_latest_checkpoint(self):
        """Return the path of the most recently saved step checkpoint."""
        return get_last_checkpoint(OUTPUT_DIR)

    def _copy_checkpoint(self, src: str, dst: str):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs
    ):
        accuracy = (metrics or {}).get("eval_accuracy", None)
        if accuracy is None:
            return

        latest_ckpt = self._get_latest_checkpoint()
        if latest_ckpt is None:
            return

        # Always update checkpoint-last
        self._copy_checkpoint(latest_ckpt, self.LAST_DIR)
        print(f"[BestCheckpointCallback] Saved checkpoint-last ← {latest_ckpt}")

        # Update checkpoint-best if improved
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self._copy_checkpoint(latest_ckpt, self.BEST_DIR)
            meta = {
                "step": state.global_step,
                "best_accuracy": self.best_accuracy,
                "source_checkpoint": latest_ckpt,
            }
            with open(self.META_FILE, "w") as f:
                json.dump(meta, f, indent=2)
            print(
                f"[BestCheckpointCallback] 🏆 New best accuracy {accuracy:.4f} — "
                f"saved checkpoint-best ← {latest_ckpt}"
            )
        else:
            print(
                f"[BestCheckpointCallback] Accuracy {accuracy:.4f} did not improve "
                f"(best={self.best_accuracy:.4f}). checkpoint-best unchanged."
            )


# ==========================================
# 7. TRAINING SETUP
# ==========================================
print("Initializing Trainer...")

gen_eval_callback = GenerationEvalCallback(
    raw_eval_dataset=raw_eval,
    tokenizer=tokenizer,
    max_new_tokens=256,
    gen_batch_size=4,  # lower if OOM; higher for speed
)
best_ckpt_callback = BestCheckpointCallback()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # NOTE: compute_metrics / preprocess_logits_for_metrics removed;
    # accuracy is now computed in GenerationEvalCallback via model.generate()
    callbacks=[
        gen_eval_callback,
        best_ckpt_callback,
        EarlyStoppingCallback(early_stopping_patience=5),
    ],
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        max_steps=5000,
        logging_steps=1,
        # --- EVAL & SAVE EVERY 100 STEPS ---
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        # --- TRACK BEST MODEL ACCORDING TO ACCURACY ---
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        # ----------------------------------------------
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,  # keep the 3 most recent step checkpoints
        report_to="none",
        dataloader_pin_memory=False,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ==========================================
# 8. TRAINING W/ RESUME
# ==========================================
print("Checking for existing checkpoints...")

last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint is not None:
    print(f"Found checkpoint at {last_checkpoint}. Resuming training...")
else:
    print("No existing checkpoint found. Starting from scratch...")

print("Starting fine-tuning...")
trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

# ==========================================
# 9. SAVE OUTPUTS
# ==========================================
print("Saving best LoRA adapters...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Merging best LoRA into base model (16-bit)...")
model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method="merged_16bit",
    maximum_memory_usage=0.7,
)
print(f"Done! Best merged model saved at: {MERGED_DIR}")
