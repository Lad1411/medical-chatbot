import argparse
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset 
# --- IMPORT MODULE RETRIEVER CỦA BẠN ---
from vector_db import build_database
from retriever import build_retriever

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "/content/drive/MyDrive/models/qwen_chatdoctor_merged"
MAX_CONTEXT_TOKENS = 2048

# ==========================================
# 2. SYSTEM PROMPTS
# ==========================================
SYS_PROMPT_NO_RAG = (
    "You are an expert medical professional. You will be provided with a medical "
    "multiple-choice question and 4 options (A, B, C, D). "
    "Read the question carefully and output ONLY the single correct option letter (A, B, C, or D). "
    "Do not provide any explanation or additional text."
)

SYS_PROMPT_RAG = (
    "You are an expert medical professional. You will be provided with reference context, "
    "a medical multiple-choice question, and 4 options (A, B, C, D). "
    "Use the provided context to determine the correct answer. "
    "Output ONLY the single correct option letter (A, B, C, or D). "
    "Do not provide any explanation or additional text."
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

def extract_answer(llm_output):
    """Sử dụng Regex để bắt chữ cái A, B, C, D từ output của LLM."""
    match = re.search(r'\b([A-D])\b', llm_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def build_messages(question, options, tokenizer, use_rag=False, raw_context=""):
    """
    Xây dựng prompt dưới dạng list of dicts.
    """
    options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
    question_block = f"Question: {question}\nOptions:\n{options_text}\nAnswer:"
    
    sys_prompt = SYS_PROMPT_RAG if use_rag else SYS_PROMPT_NO_RAG
    
    if not use_rag or not raw_context:
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question_block}
        ]
    
    # Xử lý RAG và giới hạn Token
    sys_tokens = len(tokenizer.encode(sys_prompt))
    q_tokens = len(tokenizer.encode(question_block))
    available_context_tokens = MAX_CONTEXT_TOKENS - sys_tokens - q_tokens - 50 
    
    if available_context_tokens <= 0:
        user_content = f"Context:\nNone\n\n{question_block}"
    else:
        context_tokens = tokenizer.encode(raw_context)
        if len(context_tokens) > available_context_tokens:
            truncated_context = tokenizer.decode(context_tokens[:available_context_tokens])
        else:
            truncated_context = raw_context
            
        user_content = f"Context:\n{truncated_context}\n\n{question_block}"
        
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]

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

    print("Loading LLM model and tokenizer into memory...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16, 
    )
    
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10,       
        temperature=0.01,        
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # ---------------------------------------------------------
    # LOAD DATASET TỪ HUGGING FACE
    # ---------------------------------------------------------
    print("\n[+] Loading GBaker/MedQA-USMLE-4-options dataset...")
    # Thường benchmark sẽ đánh giá trên tập "test"
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    
    # Lấy một số lượng nhỏ để test nếu truyền tham số --limit
    if args.limit > 0:
        print(f"[*] Limiting test to first {args.limit} questions.")
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        
    total = len(dataset)
    correct = 0
    
    print(f"[*] Total questions to process: {total}")
    print("\nStarting inference...\n")
    
    for idx, item in enumerate(dataset):
        # Trích xuất đúng chuẩn key từ tập GBaker/MedQA-USMLE-4-options
        question = item["question"]
        options = item["options"]
        actual = item["answer_idx"] # Tập này dùng answer_idx để lưu "A", "B", "C", hoặc "D"
        
        # 1. RETRIEVE CONTEXT
        raw_context = ""
        if args.rag and retriever_engine is not None:
            hits = retriever_engine.retrieve(question, final_top_k=3, use_rerank=True)
            raw_context = retriever_engine.format_context(hits)
            
        # 2. Build messages
        messages = build_messages(
            question, 
            options, 
            tokenizer, 
            use_rag=args.rag, 
            raw_context=raw_context
        )
        
        # 3. Apply Chat Template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 4. Generate & Extract
        outputs = text_generator(prompt)
        generated_text = outputs[0]["generated_text"][len(prompt):]
        pred = extract_answer(generated_text)
        
        # Đánh giá đúng sai
        is_correct = (pred == actual)
        if is_correct:
            correct += 1
            
        # In ra màn hình console (in đè trên 1 dòng để không bị dài màn hình nếu muốn, 
        # nhưng để dễ xem ta in từng dòng)
        print(f"Q{idx+1}/{total} | Expected: {actual} | Predicted: {pred} | Correct: {is_correct}")
        
    accuracy = (correct / total) * 100
    print(f"\n{'='*40}")
    print(f"FINAL BENCHMARK RESULTS")
    print(f"{'='*40}")
    print(f"Mode used:      {'Hybrid RAG + LLM' if args.rag else 'Zero-shot LLM (No RAG)'}")
    print(f"Total processed:{total}")
    print(f"Correct answers:{correct}")
    print(f"Accuracy:       {accuracy:.2f}%")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()