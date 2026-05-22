from unsloth import FastLanguageModel
import argparse
import csv
import re
import torch
from transformers import pipeline
from datasets import load_dataset 
from tqdm.autonotebook import tqdm

# --- IMPORT MODULE RETRIEVER CỦA BẠN ─--
from retriever import build_retriever

# ==========================================
# 1. CONFIGURATION & CORE HYPERPARAMETERS
# ==========================================
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_CONTEXT_TOKENS = 2048

# ==========================================
# 2. SYSTEM PROMPTS (ALIGNING TO TRAINING PROMPT)
# ==========================================
# System prompts aligned perfectly with the fine-tuning instruction:
# "You are a helpful and expert medical assistant. Identify the correct response employing a logical and systematic strategy."

SYS_PROMPT_MEDQA_NO_RAG = (
    "You are an expert medical assistant. Please think step by step to analyze the "
    "clinical scenario, symptoms, and potential diagnoses. Evaluate each given option "
    "logically before making a decision. You must conclude your reasoning by providing "
    "the final answer as exactly one letter: A, B, C, or D."
)

SYS_PROMPT_MEDQA_RAG = (
    "You are an expert medical assistant. Carefully read and use the provided context "
    "to inform your reasoning. Think step by step to analyze the clinical question and "
    "evaluate the given options based strictly on the retrieved medical text. You must "
    "conclude your reasoning by providing the final answer as exactly one letter: A, B, C, or D."
)

# --- PubMedQA (Yes/No/Maybe) Prompts ---

SYS_PROMPT_PUBMED_NO_RAG = (
    "You are an expert medical research assistant. Please think step by step to analyze "
    "the research question based on general medical knowledge. You must conclude your "
    "reasoning with a final classification of 'yes', 'no', or 'maybe'. "
    "Output 'yes' if the medical consensus strongly supports or confirms the premise. "
    "Output 'no' if the medical consensus contradicts or refutes the premise. "
    "Output 'maybe' if the evidence is inconclusive, highly conditional, or yields mixed results."
)

SYS_PROMPT_PUBMED_RAG = (
    "You are an expert medical research assistant. Use the provided context to guide "
    "your reasoning. Think step by step to evaluate the research question based on the "
    "findings in the retrieved documents. You must conclude your reasoning with a final "
    "classification of 'yes', 'no', or 'maybe'. "
    "Output 'yes' if the context explicitly confirms or supports the premise. "
    "Output 'no' if the context explicitly refutes or contradicts the premise. "
    "Output 'maybe' if the context states the results are inconclusive, conditional, or mixed."
)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MedQA & PubMedQA with optional hybrid RAG")
    
    # LoRA Path parameter
    parser.add_argument(
        "--lora-path", "--lora_path", 
        type=str, 
        default="./models/qwen_chatdoctor_lora_new_dataset/checkpoint-200", 
        help="Path to the LoRA adapter checkpoints"
    )
    
    # RAG flag parameter
    parser.add_argument(
        "--rag", "--use-rag", "--use_rag",
        action="store_true", 
        help="Enable RAG (Hybrid Retrieval + Reranking) mode"
    )
    
    # Output file name parameter
    parser.add_argument(
        "--output", "--output-file", "--output_file",
        type=str, 
        default="benchmark_results.csv", 
        help="Path/Name of the output CSV file to save benchmark results"
    )
    
    # Question limit parameter (for fast-testing / subset eval)
    parser.add_argument(
        "--limit", 
        type=int, 
        default=0, 
        help="Limit number of questions to test per dataset (0 for all)"
    )
    
    return parser.parse_args()

def extract_mcq_answer(llm_output):
    """Parses single character options (A, B, C, D) from LLM generation."""
    match = re.search(r'\b([A-D])\b', llm_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def extract_ynm_answer(llm_output):
    """Parses PubMedQA classification responses (yes, no, maybe) from LLM generation."""
    match = re.search(r'\b(yes|no|maybe)\b', llm_output, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None

def format_context(hits):
    """Formats list of retrieved documents into a clean structural text block."""
    if not hits:
        return "No reference context found."
        
    formatted_docs = []
    for i, hit in enumerate(hits):
        title = hit.get("metadata", {}).get("title", f"Document {i+1}")
        text = hit.get("text", "")
        formatted_docs.append(f"[Document {i+1} - Title: {title}]\n{text}")
        
    return "\n\n".join(formatted_docs)

def build_messages(user_prompt_text, sys_prompt_no_rag, sys_prompt_rag, tokenizer, use_rag=False, raw_context=""):
    """
    Constructs conversations while managing and formatting context.
    """
    sys_prompt = sys_prompt_rag if use_rag else sys_prompt_no_rag
    
    if not use_rag or not raw_context:
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_text}
        ]
    
    # Context token safety tracking
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

def run_medqa_eval(dataset, text_generator, tokenizer, retriever_engine, use_rag, results_list):
    correct = 0
    total = len(dataset)
    print(f"\n[*] Evaluating MedQA ({total} questions)...")

    progressbar = tqdm(dataset, desc="MedQA benchmark progress", colour='cyan')
    
    for idx, item in enumerate(progressbar):
        question = item["question"]
        options = item["options"]
        actual = item["answer_idx"] 
        
        raw_context = ""
        if use_rag and retriever_engine is not None:
            # Query textbook VectorDB with the clinical question
            hits = retriever_engine.retrieve(question, final_top_k=3, use_rerank=True)
            raw_context = format_context(hits)
            
        options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
        user_prompt_text = f"Question: {question}\nOptions:\n{options_text}\nAnswer:"
        
        messages = build_messages(user_prompt_text, SYS_PROMPT_MEDQA_NO_RAG, SYS_PROMPT_MEDQA_RAG, tokenizer, use_rag, raw_context)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = text_generator(prompt)
        generated_text = outputs[0]["generated_text"][len(prompt):]
        pred = extract_mcq_answer(generated_text)
        
        is_correct = (pred == actual)
        if is_correct: 
            correct += 1
            
        print(f"  MedQA Q{idx+1}/{total} | Expect: {actual} | Predict: {pred} | Correct: {is_correct}")
        
        # Append report object to final export list
        results_list.append({
            "Dataset": "MedQA",
            "Index": idx + 1,
            "Question": question,
            "Context": raw_context if use_rag else "N/A (No RAG)",
            "Expected": actual,
            "Predicted": pred if pred is not None else "[Invalid Output]",
            "IsCorrect": is_correct
        })
        
    return correct, total

def run_pubmedqa_eval(dataset, text_generator, tokenizer, retriever_engine, use_rag, results_list):
    correct = 0
    total = len(dataset)
    print(f"\n[*] Evaluating PubMedQA ({total} questions)...")

    progressbar = tqdm(dataset, desc="PubMedQA benchmark progress", colour='cyan')
        
    for idx, item in enumerate(progressbar):
        question = item["question"]
        actual = item["final_decision"] # yes, no, maybe
        
        raw_context = ""
        if use_rag and retriever_engine is not None:
            # Query textbook VectorDB with the research question
            hits = retriever_engine.retrieve(question, final_top_k=3, use_rerank=True)
            raw_context = format_context(hits)
            
        user_prompt_text = f"Question: {question}\nAnswer:"
        
        messages = build_messages(user_prompt_text, SYS_PROMPT_PUBMED_NO_RAG, SYS_PROMPT_PUBMED_RAG, tokenizer, use_rag, raw_context)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = text_generator(prompt)
        generated_text = outputs[0]["generated_text"][len(prompt):]
        pred = extract_ynm_answer(generated_text)
        
        is_correct = (pred == actual)
        if is_correct: 
            correct += 1
            
        # print(f"  PubMedQA Q{idx+1}/{total} | Expect: {actual} | Predict: {pred} | Correct: {is_correct}")
        
        # Append report object to final export list
        results_list.append({
            "Dataset": "PubMedQA",
            "Index": idx + 1,
            "Question": question,
            "Context": raw_context if use_rag else "N/A (No RAG)",
            "Expected": actual,
            "Predicted": pred if pred is not None else "[Invalid Output]",
            "IsCorrect": is_correct
        })
        
    return correct, total

# ==========================================
# 4. MAIN BENCHMARK EXECUTION
# ==========================================
def main():
    args = parse_args()
    print(f"--- Running Benchmark on MedQA & PubMedQA dataset ---")
    print(f"Mode: {'Hybrid RAG + LLM' if args.rag else 'Zero-shot LLM (No RAG)'}")
    print(f"LoRA Adapter Path: {args.lora_path}")
    print(f"Output CSV Path: {args.output}")
    if args.limit > 0:
        print(f"Question Limit: {args.limit} queries per dataset")
    
    # ---------------------------------------------------------
    # KHỞI TẠO HỆ THỐNG RAG (Chỉ khởi tạo 1 lần)
    # ---------------------------------------------------------
    retriever_engine = None
    if args.rag:
        print("\n[+] Initializing Vector Database and Hybrid Retriever...")
        retriever_engine = build_retriever()
        print("[+] RAG System Ready!\n")

    print("\n[+] Loading Base LLM (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_CONTEXT_TOKENS,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    print(f"[+] Applying LoRA adapters from {args.lora_path}...")
    model.load_adapter(args.lora_path)
    
    # BẬT CHẾ ĐỘ NATIVE INFERENCE CỦA UNSLOTH (Tốc độ x2, chống lỗi RAM)
    FastLanguageModel.for_inference(model)

    print("[+] Model is ready for inference!")

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=40,       
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # ---------------------------------------------------------
    # LOAD DATASETS TỪ HUGGING FACE
    # ---------------------------------------------------------
    print("\n[+] Loading Medical Datasets...")
    medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    pubmedqa_ds = load_dataset("pubmed_qa", "pqa_labeled", split="train") 
    
    if args.limit > 0:
        medqa_ds = medqa_ds.select(range(min(args.limit, len(medqa_ds))))
        pubmedqa_ds = pubmedqa_ds.select(range(min(args.limit, len(pubmedqa_ds))))

    # Chạy Benchmark
    results_data = [] # Stores rows for CSV output
    med_correct, med_total = run_medqa_eval(medqa_ds, text_generator, tokenizer, retriever_engine, args.rag, results_data)
    pub_correct, pub_total = run_pubmedqa_eval(pubmedqa_ds, text_generator, tokenizer, retriever_engine, args.rag, results_data)
    
    # In báo cáo tổng hợp
    print(f"\n{'='*50}")
    print(f"FINAL BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Mode used: {'Hybrid RAG + LLM' if args.rag else 'Zero-shot LLM (No RAG)'}")
    print(f"{'-'*50}")
    print(f"[1] MedQA (USMLE 4-options):")
    print(f"    - Correct: {med_correct} / {med_total}")
    print(f"    - Accuracy: {(med_correct / med_total * 100) if med_total else 0:.2f}%")
    print(f"[2] PubMedQA (Yes/No/Maybe):")
    print(f"    - Correct: {pub_correct} / {pub_total}")
    print(f"    - Accuracy: {(pub_correct / pub_total * 100) if pub_total else 0:.2f}%")
    print(f"{'='*50}")

    # Ghi toàn bộ kết quả vào File CSV
    print(f"\n[+] Writing benchmark metrics to: {args.output}...")
    try:
        med_acc = (med_correct / med_total * 100) if med_total else 0.0
        pub_acc = (pub_correct / pub_total * 100) if pub_total else 0.0
        
        with open(args.output, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Dataset", "Correct", "Total", "Accuracy"])
            writer.writerow(["MedQA (USMLE 4-options)", med_correct, med_total, f"{med_acc:.2f}%"])
            writer.writerow(["PubMedQA (Yes/No/Maybe)", pub_correct, pub_total, f"{pub_acc:.2f}%"])
            
        print(f"[+] Successfully exported final results to {args.output}!")
    except Exception as e:
        print(f"[-] Failed to export results to CSV: {e}")

if __name__ == "__main__":
    main()
