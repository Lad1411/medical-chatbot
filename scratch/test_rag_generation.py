import sys
import os
import time

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from datasets import load_dataset
from retriever import build_retriever

print("[+] Initializing retriever (without building BM25)...")
from vector_db import VectorDB
from retriever import HybridRetriever

db = VectorDB()
retriever = HybridRetriever(db) # Bypasses build_bm25_index() entirely!

print("[+] Loading MedQA-Mixtral-CoT dataset...")
dataset = load_dataset("HPAI-BSC/MedQA-Mixtral-CoT", split="train")

print(f"[+] Total samples in dataset: {len(dataset)}")

# Test on first 100 samples
n_samples = 100
print(f"[+] Retrieving context (DENSE ONLY) for the first {n_samples} samples...")

start_time = time.time()
for i in range(n_samples):
    question = dataset[i]["question"]
    # Retrieve top 3 documents using dense search only
    hits = retriever._dense_search(question, top_k=3)
    if i > 0 and i % 20 == 0:
        print(f" - Processed {i}/{n_samples}...")

end_time = time.time()
avg_time = (end_time - start_time) / n_samples
total_estimated_time = avg_time * len(dataset)

print(f"\n[+] Done in {end_time - start_time:.2f} seconds.")
print(f"[+] Average time per query: {avg_time:.4f} seconds.")
print(f"[+] Estimated time for all {len(dataset)} samples: {total_estimated_time/60:.2f} minutes.")
