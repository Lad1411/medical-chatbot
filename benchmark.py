from unsloth import FastLanguageModel
import argparse
import csv
import re
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import logging

# ==========================================
# PERFORMANCE SETTINGS
# ==========================================
torch.backends.cuda.matmul.allow_tf32 = True
logging.set_verbosity_error()

# ==========================================
# CONFIGURATION
# ==========================================
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_CONTEXT_TOKENS = 1536
MAX_NEW_TOKENS = 256
BATCH_SIZE = 4

# ==========================================
# OPTIONAL RAG IMPORT
# ==========================================
# from retriever import build_retriever

# ==========================================
# SYSTEM PROMPTS
# ==========================================
SYS_PROMPT_MEDQA_NO_RAG = (
    "You are a helpful and expert medical assistant. Identify the correct response employing a logical and systematic strategy. "
    "Evaluate each option logically and conclude your reasoning with a final answer as exactly one letter: A, B, C, or D."
)

SYS_PROMPT_MEDQA_RAG = (
    "You are a helpful and expert medical assistant. Identify the correct response employing a logical and systematic strategy. "
    "Carefully use the provided medical context to inform your step-by-step reasoning. Conclude your reasoning with a final answer as exactly one letter: A, B, C, or D."
)

# --- PubMedQA (Yes/No/Maybe) Prompts ---
SYS_PROMPT_PUBMED_NO_RAG = (
    "You are a helpful and expert medical assistant. Identify the correct response employing a logical and systematic strategy. "
    "Evaluate the medical premise and conclude your reasoning with a final classification of 'yes', 'no', or 'maybe'."
)

SYS_PROMPT_PUBMED_RAG = (
    "You are a helpful and expert medical assistant. Identify the correct response employing a logical and systematic strategy. "
    "Carefully use the provided research context to evaluate the question. Conclude your reasoning with a final classification of 'yes', 'no', or 'maybe'."
)

# ==========================================
# ARGUMENT PARSER
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", type=str, default="./models/checkpoint-1600")
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=str, default="benchmark_results.csv")
    return parser.parse_args([])

# ==========================================
# ANSWER EXTRACTION
# ==========================================
# def extract_mcq_answer(text):
#     patterns = [
#         r"Final Answer:\s*([A-D])",
#         r"Answer:\s*([A-D])",
#         r"answer is\s*([A-D])",
#         r"\b([A-D])\b\s*$",
#     ]
#     for pattern in patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             return match.group(1).upper()
#     return None

def extract_mcq_answer(llm_output):
    """Parses single character options (A, B, C, D) from LLM generation."""
    match_mcq = re.search(r"(?:Answer:|answer is|correct option is|choice is)\s*([A-D])", llm_output, re.IGNORECASE)
    if match_mcq:
        return match_mcq.group(1).upper()

    match_mcq_end = re.search(r"\b([A-D])\b[\.\s]*(?:<\|im_end\|>)?$", llm_output, re.IGNORECASE)
    if match_mcq_end:
        return match_mcq_end.group(1).upper()
    return None

def extract_ynm_answer(llm_output):
    """Parses PubMedQA classification responses (yes, no, maybe) from LLM generation."""
    match_ynm = re.search(r"(?:Answer:|answer is|correct option is|choice is)\s*(yes|no|maybe)", llm_output, re.IGNORECASE)
    if match_ynm:
        return match_ynm.group(1).lower()

    match_ynm = re.search(r"\b(yes|no|maybe)\b[\.\s]*(?:<\|im_end\|>)?$", llm_output, re.IGNORECASE)
    if match_ynm:
        return match_ynm.group(1).lower()
    return None


# ==========================================
# CONTEXT FORMATTER
# ==========================================
def format_context(hits):
    if not hits:
        return "No reference context found."
    
    docs = []
    for i, hit in enumerate(hits):
        title = hit.get("metadata", {}).get("title", f"Document {i+1}")
        text = hit.get("text", "")
        docs.append(f"[Document {i+1} - {title}]\n{text}")
        
    return "\n\n".join(docs)

# ==========================================
# BUILD MESSAGES
# ==========================================
def build_messages(user_prompt_text, sys_prompt_no_rag, sys_prompt_rag, tokenizer, use_rag=False, raw_context=""):
    sys_prompt = sys_prompt_rag if use_rag else sys_prompt_no_rag

    if not use_rag or not raw_context:
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_text}
        ]

    sys_tokens = len(tokenizer.encode(sys_prompt))
    q_tokens = len(tokenizer.encode(user_prompt_text))
    available_context_tokens = MAX_CONTEXT_TOKENS - sys_tokens - q_tokens - 50

    if available_context_tokens <= 0:
        user_content = f"Context:\nNone\n\n{user_prompt_text}"
    else:
        context_tokens = tokenizer.encode(raw_context)
        if len(context_tokens) > available_context_tokens:
            truncated_context = tokenizer.decode(context_tokens[:available_context_tokens])
        else:
            truncated_context = raw_context
            
        user_content = f"Context:\n{truncated_context}\n\n{user_prompt_text}"

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]

# ==========================================
# BATCH GENERATION
# ==========================================
def generate_batch(model, tokenizer, prompts):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1] 

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.05,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice out only the new tokens generated
    generated_tokens = outputs[:, input_length:] 
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    return [text.strip() for text in decoded]

# ==========================================
# MEDQA EVALUATION
# ==========================================
def run_medqa_eval(dataset, model, tokenizer, retriever_engine, use_rag, results_list):
    correct = 0
    total = len(dataset)
    print(f"\n[*] Evaluating MedQA ({total} questions)...")
    progressbar = tqdm(total=total, desc="MedQA", colour="cyan")

    for start_idx in range(0, total, BATCH_SIZE):
        batch = dataset[start_idx:start_idx + BATCH_SIZE]
        prompts, actual_answers, questions, contexts = [], [], [], []


        for question, options, actual in zip(batch["question"], batch["options"], batch["answer_idx"]):
            raw_context = ""

            if use_rag and retriever_engine is not None:
                hits = retriever_engine.retrieve(question, final_top_k=3, use_rerank=True)
                raw_context = format_context(hits)

            options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
            user_prompt_text = f"Question: {question}\nOptions:\n{options_text}\nAnswer:"

            messages = build_messages(user_prompt_text, SYS_PROMPT_MEDQA_NO_RAG, SYS_PROMPT_MEDQA_RAG, tokenizer, use_rag, raw_context)

            # print(messages)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # print(prompt)

            # exit()

            prompts.append(prompt)
            actual_answers.append(actual)
            questions.append(question)
            contexts.append(raw_context)

        generated_outputs = generate_batch(model, tokenizer, prompts)

        for i, generated_text in enumerate(generated_outputs):
            pred = extract_mcq_answer(generated_text)
            is_correct = (pred == actual_answers[i].lower())
            
            if is_correct:
                correct += 1

            results_list.append({
                "Dataset": "MedQA",
                "Index": start_idx + i + 1,
                "Question": questions[i],
                "Context": contexts[i] if use_rag else "N/A",
                "Expected": actual_answers[i],
                "Predicted": pred if pred else "[Invalid]",
                "IsCorrect": is_correct
            })

        current_batch_len = len(batch["question"])
        progressbar.update(current_batch_len)
    progressbar.close()
    
    return correct, total

# ==========================================
# PUBMEDQA EVALUATION
# ==========================================
def run_pubmedqa_eval(dataset, model, tokenizer, retriever_engine, use_rag, results_list):
    correct = 0
    total = len(dataset)
    print(f"\n[*] Evaluating PubMedQA ({total} questions)...")
    progressbar = tqdm(total=total, desc="PubMedQA", colour="green")

    for start_idx in range(0, total, BATCH_SIZE):
        batch = dataset[start_idx:start_idx + BATCH_SIZE]
        prompts, actual_answers, questions, contexts = [], [], [], []

        # print(batch)

        for question, actual in zip(batch["question"], batch["final_decision"]):
            raw_context = ""

            if use_rag and retriever_engine is not None:
                hits = retriever_engine.retrieve(question, final_top_k=3, use_rerank=True)
                raw_context = format_context(hits)

            user_prompt_text = f"Question: {question}\nAnswer:"
            messages = build_messages(user_prompt_text, SYS_PROMPT_PUBMED_NO_RAG, SYS_PROMPT_PUBMED_RAG, tokenizer, use_rag, raw_context)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # print(prompt)
            # exit()

            prompts.append(prompt)
            actual_answers.append(actual)
            questions.append(question)
            contexts.append(raw_context)

        generated_outputs = generate_batch(model, tokenizer, prompts)

        for i, generated_text in enumerate(generated_outputs):
            pred = extract_ynm_answer(generated_text)
            is_correct = (pred == actual_answers[i])
            
            if is_correct:
                correct += 1

            results_list.append({
                "Dataset": "PubMedQA",
                "Index": start_idx + i + 1,
                "Question": questions[i],
                "Context": contexts[i] if use_rag else "N/A",
                "Expected": actual_answers[i],
                "Predicted": pred if pred else "[Invalid]",
                "IsCorrect": is_correct
            })

        current_batch_len = len(batch["question"])
        progressbar.update(current_batch_len)
    progressbar.close()
    
    return correct, total

# ==========================================
# MAIN
# ==========================================
def main():
    args = parse_args()

    print("=" * 60)
    print("MEDICAL BENCHMARK")
    print("=" * 60)

    retriever_engine = None
    if args.rag:
        print("[+] Initializing RAG system...")
        retriever_engine = build_retriever()
        print("[+] RAG ready.")

    print("\n[+] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_CONTEXT_TOKENS,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer.padding_side = "left"

    print(f"[+] Loading LoRA: {args.lora_path}")
    model.load_adapter(args.lora_path)
    FastLanguageModel.for_inference(model)
    model.eval()
    print("[+] Model ready.")

    print("\n[+] Loading datasets...")
    medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    pubmedqa_ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    if args.limit > 0:
        medqa_ds = medqa_ds.select(range(min(args.limit, len(medqa_ds))))
        pubmedqa_ds = pubmedqa_ds.select(range(min(args.limit, len(pubmedqa_ds))))

    results_data = []
    
    med_correct, med_total = run_medqa_eval(medqa_ds, model, tokenizer, retriever_engine, args.rag, results_data)
    pub_correct, pub_total = run_pubmedqa_eval(pubmedqa_ds, model, tokenizer, retriever_engine, args.rag, results_data)

    med_acc = (med_correct / med_total * 100) if med_total > 0 else 0
    pub_acc = (pub_correct / pub_total * 100) if pub_total > 0 else 0

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"MedQA Accuracy: {med_acc:.2f}%")
    print(f"PubMedQA Accuracy: {pub_acc:.2f}%")
    print("=" * 60)

    print(f"\n[+] Writing results to {args.output}")
    with open(args.output, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Dataset", "Correct", "Total", "Accuracy"])
        writer.writerow(["MedQA", med_correct, med_total, f"{med_acc:.2f}%"])
        writer.writerow(["PubMedQA", pub_correct, pub_total, f"{pub_acc:.2f}%"])

    print("[+] Done.")

if __name__ == "__main__":
    main()