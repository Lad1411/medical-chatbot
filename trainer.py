import os
import re
import numpy as np
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainingArguments, EarlyStoppingCallback


# ==========================================
# 1. CONFIGURATION & KAGGLE PATHS
# ==========================================
max_seq_length = 1536            # Maximum sequence length for training
dtype = None                     # Auto-detect precision (fp16/bf16)
load_in_4bit = True              # Enable 4-bit quantization (QLoRA)

OUTPUT_DIR = "./models/qwen_chatdoctor_lora_new_dataset"
MERGED_DIR = "./models/qwen_chatdoctor_merged"

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
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ==========================================
# 3. PREPARE DATASET
# ==========================================
print("Loading MedQA-Mixtral-CoT dataset...")

dataset = load_dataset("HPAI-BSC/MedQA-Mixtral-CoT", split="train")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

EOS_TOKEN = tokenizer.eos_token

def filter_by_response_length(example):
    response_text = example["response"]
    question_text = example["question"]
        
    tokenized_resp = tokenizer(str(response_text), truncation=False, add_special_tokens=False)
    tokenized_quest = tokenizer(str(question_text), truncation=False, add_special_tokens=False)

    total_length = len(tokenized_resp["input_ids"]) + len(tokenized_quest["input_ids"])
    return total_length < max_seq_length - 10 - 25

print("Filtering dataset...")
dataset = dataset.filter(filter_by_response_length, num_proc=2)

def formatting_prompts_func(examples):
    inputs = examples["question"] 
    outputs = examples["response"]
    texts = []

    for input_text, output in zip(inputs, outputs):
        messages = [
            {"role": "system",    "content": "You are a helpful and expert medical assistant. Identify the correct response employing a logical and systematic strategy."},
            {"role": "user",      "content": input_text},
            {"role": "assistant", "content": str(output)},
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

print("Splitting dataset into train and eval...")
split_dataset = dataset.train_test_split(test_size=200, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ==========================================
# 4. CUSTOM EVALUATION METRICS
# ==========================================
def extract_answer(text):
    """
    Robust parser to extract multiple choice answers (A, B, C, D).
    """
    # Strategy 1: Look for common concluding phrases
    # Matches: "Answer: A", "The answer is B", "correct option is C"
    match = re.search(r"(?:Answer:|answer is|correct option is|choice is)\s*([A-D])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Strategy 2: Fallback to the very end of the generated text
    # Matches a standalone A, B, C, or D right before the end token or period
    # e.g., "Therefore, A.<|im_end|>" or just "B"
    match_end = re.search(r"\b([A-D])\b[\.\s]*(?:<\|im_end\|>)?$", text, re.IGNORECASE)
    if match_end:
        return match_end.group(1).upper()
        
    # If the model completely fails to provide a recognizable answer
    return None

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    
    correct = 0
    total = 0
    for pred_text, label_text in zip(decoded_preds, decoded_labels):
        pred_ans = extract_answer(pred_text)
        label_ans = extract_answer(label_text)
        if label_ans is not None:
            total += 1
            if pred_ans == label_ans:
                correct += 1
                
    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy}

# ==========================================
# 5. TRAINING SETUP
# ==========================================
print("Initializing Trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    
    # Link the custom metrics and preprocessing
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,

    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],

    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,

        per_device_train_batch_size=8,
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
        greater_is_better=True, # Accuracy should maximize, unlike loss
        # ----------------------------------------------

        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,

        output_dir=OUTPUT_DIR,
        save_total_limit=2,        

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
# 6. TRAINING W/ RESUME
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
# 7. SAVE OUTPUTS
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