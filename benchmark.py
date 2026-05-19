import argparse
import json
import re
import torch
from transformers import pipeline
from datasets import load_dataset 
from unsloth import FastLanguageModel

# --- IMPORT MODULE RETRIEVER CỦA BẠN ---
# from vector_db import build_database
# from retriever import build_retriever
from peft import PeftModel

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
LORA_PATH = "./models/qwen_chatdoctor_lora_new_dataset/checkpoint-200" 
MAX_CONTEXT_TOKENS = 2048

# ==========================================
# 2. SYSTEM PROMPTS
# ==========================================
# SYS_PROMPT_MEDQA_NO_RAG = (
#     """
#     You are a helpful and expert medical assistant. Identify the correct response employing a logical and systematic strategy.
#     """
# )

# SYS_PROMPT_MEDQA_RAG = (
#     "You are an expert medical professional. You will be provided with reference context, "
#     "a medical multiple-choice question, and 4 options (A, B, C, D). "
#     "Use the provided context to determine the correct answer. "
#     "Output ONLY the single correct option letter (A, B, C, or D). "
#     "Do not provide any explanation or additional text."
# )

# # --- PUBMEDQA PROMPTS ---
# SYS_PROMPT_PUBMED_NO_RAG = (
#     """
#     You are an expert medical researcher. You will be provided with a medical research question.
#     First, briefly analyze the established medical consensus regarding this topic.
#     Then, conclude your response with ONLY one of the following words wrapped in brackets: [yes], [no], or [maybe].
#     - [yes]: medical knowledge strongly supports the premise.
#     - [no]: medical knowledge refutes the premise.
#     - [maybe]: evidence is inconclusive or conflicting.
#     """
# )

# SYS_PROMPT_PUBMED_RAG = (
#     "You are an expert medical researcher. You will be provided with reference context and a medical research question. "
#     "Use ONLY the provided context to answer the question with ONLY one of the following words: 'yes', 'no', or 'maybe'. "
#     "Follow these criteria strictly:\n"
#     "- Output 'yes' if the context explicitly supports the premise or shows a positive conclusion.\n"
#     "- Output 'no' if the context explicitly refutes the premise or shows a negative conclusion.\n"
#     "- Output 'maybe' if the context is inconclusive, states that more research is needed, or lacks sufficient information.\n"
#     "Do not provide any explanation or additional text."
# )

SYS_PROMPT = (
    "You are a helpful and expert medical assistant. Identify the correct response employing a logical and systematic strategy."
)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MedQA with optional RAG")
    parser.add_argument("--rag", action="store_true", help="Enable RAG (Hybrid Retrieval) mode")
    # THÊM THAM SỐ LIMIT ĐỂ TEST NHANH
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions to test (0 for all)")
    return parser.parse_args()

def extract_mcq_answer(llm_output):
    """Bắt chữ cái A, B, C, D cho MedQA."""
    match = re.search(r'\b([A-D])\b', llm_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def extract_ynm_answer(llm_output):
    """Bắt chữ yes, no, maybe cho PubMedQA."""
    match = re.search(r'\b(yes|no|maybe)\b', llm_output, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None

def build_messages(user_prompt_text, sys_prompt_no_rag, sys_prompt_rag, tokenizer, use_rag=False, raw_context=""):
    """
    Hàm chung để xây dựng message và cắt Token Context an toàn.
    """
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

def run_medqa_eval(dataset, text_generator, tokenizer, retriever_engine, use_rag):
    correct = 0
    total = len(dataset)
    print(f"\n[*] Evaluating MedQA ({total} questions)...")
    
    for idx, item in enumerate(dataset):
        question = item["question"]
        options = item["options"]
        actual = item["answer_idx"] 
        
        raw_context = ""
        if use_rag and retriever_engine is not None:
            hits = retriever_engine.retrieve(question, final_top_k=3, use_rerank=True)
            raw_context = retriever_engine.format_context(hits)
            
        options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
        user_prompt_text = f"Question: {question}\nOptions:\n{options_text}\nAnswer:"
        
        messages = build_messages(user_prompt_text, SYS_PROMPT_MEDQA_NO_RAG, SYS_PROMPT_MEDQA_RAG, tokenizer, use_rag, raw_context)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = text_generator(prompt)
        generated_text = outputs[0]["generated_text"][len(prompt):]
        pred = extract_mcq_answer(generated_text)
        
        is_correct = (pred == actual)
        if is_correct: correct += 1
            
        print(f"  MedQA Q{idx+1}/{total} | Expect: {actual} | Predict: {pred} | Correct: {is_correct}")
        
    return correct, total

def run_pubmedqa_eval(dataset, text_generator, tokenizer, retriever_engine, use_rag):
    correct = 0
    total = len(dataset)
    print(f"\n[*] Evaluating PubMedQA ({total} questions)...")
    
    for idx, item in enumerate(dataset):
        question = item["question"]
        actual = item["final_decision"] # yes, no, maybe
        
        raw_context = ""
        if use_rag and retriever_engine is not None:
            hits = retriever_engine.retrieve(question, final_top_k=3, use_rerank=True)
            raw_context = retriever_engine.format_context(hits)
            
        user_prompt_text = f"Question: {question}\nAnswer:"
        
        messages = build_messages(user_prompt_text, SYS_PROMPT_PUBMED_NO_RAG, SYS_PROMPT_PUBMED_RAG, tokenizer, use_rag, raw_context)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = text_generator(prompt)
        generated_text = outputs[0]["generated_text"][len(prompt):]
        pred = extract_ynm_answer(generated_text)
        
        is_correct = (pred == actual)
        if is_correct: correct += 1
            
        print(f"  PubMedQA Q{idx+1}/{total} | Expect: {actual} | Predict: {pred} | Correct: {is_correct}")
        
    return correct, total

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    args = parse_args()
    print(f"--- Running Benchmark on MedQA ---")
    print(f"Mode: {'Hybrid RAG + LLM' if args.rag else 'Zero-shot LLM (No RAG)'}")
    
    # ---------------------------------------------------------
    # KHỞI TẠO HỆ THỐNG RAG (Chỉ khởi tạo 1 lần)
    # ---------------------------------------------------------
    retriever_engine = None
    if args.rag:
        print("\n[+] Initializing Vector Database and Hybrid Retriever...")
        db = build_database(force_rebuild=False)
        retriever_engine = build_retriever(db)
        print("[+] RAG System Ready!\n")

    print("\n[+] Loading Base LLM (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_CONTEXT_TOKENS,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    print(f"[+] Applying LoRA adapters from {LORA_PATH}...")
    model.load_adapter(LORA_PATH)
    
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
    # LOAD DATASET TỪ HUGGING FACE
    # ---------------------------------------------------------
    # Load dữ liệu MedQA
    print("\n[+] Loading Datasets...")
    medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    
    # Load dữ liệu PubMedQA (Tập pqa_labeled chứa 500 câu hỏi chất lượng cao)
    pubmedqa_ds = load_dataset("pubmed_qa", "pqa_labeled", split="train") 
    
    if args.limit > 0:
        medqa_ds = medqa_ds.select(range(min(args.limit, len(medqa_ds))))
        pubmedqa_ds = pubmedqa_ds.select(range(min(args.limit, len(pubmedqa_ds))))

    # Chạy Benchmark
    med_correct, med_total = run_medqa_eval(medqa_ds, text_generator, tokenizer, retriever_engine, args.rag)
    pub_correct, pub_total = run_pubmedqa_eval(pubmedqa_ds, text_generator, tokenizer, retriever_engine, args.rag)
    
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

if __name__ == "__main__":
    main()