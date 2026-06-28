import sys
import os
import argparse
import time
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.proposed.retriever import build_retriever

class BioMistralPipeline:
    def __init__(self, retriever_type: str = "none", model_id: str = "ZiweiChen/BioMistral-Clinical-7B"):
        print(f"[+] Initializing BioMistral Pipeline (Mode: {retriever_type})")
        self.retriever_type = retriever_type
        if retriever_type != "none":
            self.retriever = build_retriever()
        else:
            self.retriever = None
            
        print(f"[+] Loading Model: {model_id}")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto"
        )
        self.model.eval()

    def run(self, query: str, options: dict = None, dataset_type: str = "medqa") -> tuple:
        if dataset_type == "medqa" and options:
            options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
            full_query = f"{query}\n{options_text}"
        else:
            full_query = query

        context_docs = []
        if self.retriever_type == "bm25":
            context_docs = self.retriever._bm25_search(full_query, top_k=3)
        elif self.retriever_type == "dense":
            context_docs = self.retriever._dense_search(full_query, top_k=3)
        elif self.retriever_type == "hybrid":
            context_docs = self.retriever.retrieve(full_query, final_top_k=3, use_rerank=False)

        # Prompt format
        context_str = "\n\n".join(context_docs)
        if context_str:
            prompt = f"Context:\n{context_str}\n\nQuestion:\n{full_query}\n\nAnswer concisely:"
        else:
            prompt = f"Question:\n{full_query}\n\nAnswer concisely:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=128, 
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        end_time = time.time()
        
        generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        latency_ms_per_token = ((end_time - start_time) * 1000) / max(generated_tokens, 1)
        
        pred_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return pred_text, latency_ms_per_token

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="none", choices=["none", "bm25", "dense", "hybrid"])
    parser.add_argument("--test-latency", action="store_true", help="Run a quick latency test")
    args = parser.parse_args()

    pipeline = BioMistralPipeline(retriever_type=args.mode)
    
    if args.test_latency:
        print("\n[+] Running Latency Test...")
        query = "A 65-year-old man presents with progressive shortness of breath and a chronic cough. What is the most likely diagnosis?"
        options = {"A": "Asthma", "B": "COPD", "C": "Pneumonia", "D": "Heart Failure"}
        
        # Warmup
        print("[+] Warmup run...")
        pipeline.run(query, options)
        
        # Actual test
        latencies = []
        for i in range(3):
            print(f"[+] Test run {i+1}...")
            _, lat = pipeline.run(query, options)
            latencies.append(lat)
            
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n[+] Average Latency: {avg_latency:.2f} ms/token")
        
if __name__ == "__main__":
    main()
