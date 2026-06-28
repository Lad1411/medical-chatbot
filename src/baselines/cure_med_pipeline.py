import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def extract_mcq_answer(text):
    """
    Extracts A, B, C, or D from the model's output text.
    Fallback to 'A' if no valid option is found.
    """
    for char in text:
        if char.upper() in ['A', 'B', 'C', 'D']:
            return char.upper()
    return "A"

def parse_args():
    parser = argparse.ArgumentParser(description="Run MedQA benchmark for Cure-Med models with CPU offloading.")
    parser.add_argument("--model", type=str, choices=["cure-med-14b", "cure-med-32b"], required=True, help="Which model to run.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of questions for quick testing (0 = run all).")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Replace these with the actual HuggingFace repository IDs for the models
    if args.model == "cure-med-14b":
        model_id = "cure-ai/Cure-Med-14B"
    else:
        model_id = "cure-ai/Cure-Med-32B"
        
    print(f"[+] Initializing {model_id}...")
    print("[+] Using device_map='auto' for CPU offloading. This will automatically split the model across GPU and CPU RAM.")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Load model with automatic device mapping (CPU offloading)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    print("[+] Loading MedQA dataset...")
    medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    if args.limit > 0:
        medqa_ds = medqa_ds.select(range(min(args.limit, len(medqa_ds))))
        
    correct = 0
    total = len(medqa_ds)
    total_time = 0.0
    total_tokens = 0
    
    for i, row in enumerate(tqdm(medqa_ds, desc=f"Evaluating {args.model}")):
        options_text = "\n".join([f"{k}) {v}" for k, v in (row["options"] or {}).items()])
        prompt = f"Question: {row['question']}\nOptions:\n{options_text}\nPlease output only the correct option letter (A, B, C, or D)."
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,  # Greedy decoding for MCQs
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        total_time += (end_time - start_time)
        total_tokens += gen_tokens
        
        # Decode only the newly generated tokens
        pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_mcq_answer(pred_text)
        
        if str(pred).upper() == str(row["answer_idx"]).upper():
            correct += 1
            
    acc = correct / total * 100 if total > 0 else 0
    avg_throughput = total_tokens / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 50)
    print(f"  RESULTS: {args.model.upper()}")
    print("=" * 50)
    print(f"Accuracy:           {acc:.2f}% ({correct}/{total})")
    print(f"Average Throughput: {avg_throughput:.2f} tok/s")
    print(f"Note: Throughput includes CPU offloading overhead.")
    print("=" * 50)

if __name__ == "__main__":
    main()
