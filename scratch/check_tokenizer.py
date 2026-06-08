from unsloth import FastLanguageModel

BASE_MODEL = "models/qwen_phase1_merged"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=1536,
    load_in_4bit=True,
)

print("eos_token:", tokenizer.eos_token)
print("eos_token_id:", tokenizer.eos_token_id)
print("pad_token:", tokenizer.pad_token)
print("pad_token_id:", tokenizer.pad_token_id)
print("all_special_tokens:", tokenizer.all_special_tokens)
print("all_special_ids:", tokenizer.all_special_ids)
print("vocab size:", len(tokenizer))
