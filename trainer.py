import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

# ==========================================
# 1. CONFIGURATION & KAGGLE PATHS
# ==========================================
max_seq_length = 1024            # Maximum sequence length for training
dtype = None                     # Auto-detect precision (fp16/bf16)
load_in_4bit = True              # Enable 4-bit quantization (QLoRA)

# Output directories (must be inside /kaggle/working)
OUTPUT_DIR = "./models/qwen_chatdoctor_lora_new_dataset"
MERGED_DIR = "./models/qwen_chatdoctor_merged"

# Create directories if they do not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

# ==========================================
# 2. LOAD MODEL & TOKENIZER
# ==========================================
print("Loading model and tokenizer...")

# Load Qwen2.5-7B model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply LoRA (QLoRA setup) for efficient fine-tuning
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

# --- MODIFIED: Load the MedQA-Mixtral-CoT dataset ---
dataset = load_dataset("HPAI-BSC/MedQA-Mixtral-CoT", split="train")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

EOS_TOKEN = tokenizer.eos_token

# --- ADDED: Filter function to remove responses >= 2048 tokens ---
def filter_by_response_length(example):
    response_text = example["response"]
        
    tokenized = tokenizer(
        str(response_text), 
        truncation=False, 
        add_special_tokens=False
    )
    # Keep row if the response is strictly less than 2048 tokens
    return len(tokenized["input_ids"]) < 1024-10-25

print("Filtering dataset to responses < 1024 tokens...")
dataset = dataset.filter(filter_by_response_length, num_proc=2)
# -----------------------------------------------------------------

def formatting_prompts_func(examples):
    keys = examples.keys()
    
    # Dynamically fetch columns regardless of whether they use question/answer or instruction/output
    inputs       = examples["question"] 
    outputs      = examples["response"]
    
    texts = []

    for input_text, output in zip(inputs, outputs):
        user_content = input_text

        messages = [
            {"role": "system",    "content": "You are a helpful and expert medical assistant. Identify the correct response employing a logical and systematic strategy."},
            {"role": "user",      "content": user_content},
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

# Apply formatting
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=5,
)

print("Splitting dataset into train and eval...")
split_dataset = dataset.train_test_split(test_size=500, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

# ==========================================
# 4. TRAINING SETUP
# ==========================================
print("Initializing Trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,

        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=10,

        max_steps=5000,
        logging_steps=1,

        # --- THAY ĐỔI: Tính loss và lưu weights mỗi 500 steps ---
        eval_strategy="steps",
        eval_steps=100,

        save_strategy="steps",
        save_steps=100,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # --------------------------------------------------------

        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,

        output_dir=OUTPUT_DIR,
        save_total_limit=2,        

        report_to="none",
        dataloader_pin_memory=False,
    ),
)

# ==========================================
# 5. TRAINING VỚI TÍNH NĂNG RESUME
# ==========================================
print("Checking for existing checkpoints...")

# --- THÊM TÍNH NĂNG RESUME TỪ CHECKPOINT ---
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint is not None:
    print(f"Found checkpoint at {last_checkpoint}. Resuming training...")
else:
    print("No existing checkpoint found. Starting from scratch...")
# --------------------------------------------

print("Starting fine-tuning...")

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_memory = round(gpu_stats.total_memory / 1024**3, 3)

print(f"GPU: {gpu_stats.name} | Total VRAM: {max_memory} GB | Reserved: {start_gpu_memory} GB")

# Start training (truyền tham số resume_from_checkpoint vào đây)
trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

print(f"\nTraining completed!")
print(f"Peak VRAM used: {used_memory} GB")
print(f"Training time: {round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes")

# ==========================================
# 6. SAVE LORA ADAPTERS
# ==========================================
print("Saving best LoRA adapters...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Best LoRA adapters saved at: {OUTPUT_DIR}")

# ==========================================
# 7. MERGE & SAVE FULL MODEL
# ==========================================
print("Merging best LoRA into base model (16-bit)...")

model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method="merged_16bit",
    maximum_memory_usage=0.7,
)

print(f"Done! Best merged model saved at: {MERGED_DIR}")