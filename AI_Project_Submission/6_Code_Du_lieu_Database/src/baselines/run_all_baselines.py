import argparse
import sys
import os
import csv
from datasets import load_dataset
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.baselines.medrag_pipeline import MedRAGPipeline, extract_mcq_answer
from src.baselines.medgraphrag_pipeline import MedGraphRAGPipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()

    print("[+] Loading MedQA dataset...")
    medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    if args.limit > 0:
        medqa_ds = medqa_ds.select(range(min(args.limit, len(medqa_ds))))

    summaries = []

    print("\n[!] Note: SOTA baseline numbers (e.g., GPT-4, LLaMA-2 70B, CURE, MedGraphRAG 2025) are now strictly cited from literature.")
    print("[!] This script only executes the local Qwen 2.5-7B ablation pipelines for Table 2.\n")

    # MedRAG Ablations (Qwen 2.5-7B Base)
    medrag_phases = [
        {"name": "MedRAG - No RAG", "mode": "none"},
        {"name": "MedRAG - BM25 Only", "mode": "bm25"},
        {"name": "MedRAG - Dense Only", "mode": "dense"},
        {"name": "MedRAG - Hybrid (RRF-2)", "mode": "hybrid"},
    ]

    for phase in medrag_phases:
        print(f"\n[*] Running {phase['name']}...")
        pipeline = MedRAGPipeline(retriever_type=phase["mode"])
        correct = 0
        total = len(medqa_ds)
        total_throughput = 0.0

        for i, row in enumerate(tqdm(medqa_ds, desc=phase["name"])):
            pred_text, thr = pipeline.run(
                row["question"], options=row["options"], dataset_type="medqa"
            )
            pred = extract_mcq_answer(pred_text)
            total_throughput += thr
            if str(pred).upper() == str(row["answer_idx"]).upper():
                correct += 1

        acc = correct / total * 100 if total > 0 else 0
        avg_throughput = total_throughput / total if total > 0 else 0
        summaries.append(
            {
                "Baseline": "MedRAG Ablation",
                "Phase": phase["name"],
                "Correct": correct,
                "Total": total,
                "Accuracy": f"{acc:.2f}%",
                "Throughput": f"{avg_throughput:.2f} tok/s"
            }
        )

    # MedGraphRAG Ablations (Qwen 2.5-7B Base)
    medgraph_phases = [
        {"name": "GraphRAG Baseline", "mode": "baseline"},
        {"name": "+ Med-MetaGraph", "mode": "metagraph"},
        {"name": "+ Triple Graph", "mode": "triplegraph"},
        {"name": "+ U-Retrieval (Full)", "mode": "full"},
    ]

    for phase in medgraph_phases:
        print(f"\n[*] Running {phase['name']}...")
        graph_pipeline = MedGraphRAGPipeline(mode=phase["mode"])
        correct = 0
        total = len(medqa_ds)
        total_throughput = 0.0

        for i, row in enumerate(tqdm(medqa_ds, desc=phase["name"])):
            pred_text, thr = graph_pipeline.run(
                row["question"], options=row["options"], dataset_type="medqa"
            )
            pred = extract_mcq_answer(pred_text)
            total_throughput += thr
            if str(pred).upper() == str(row["answer_idx"]).upper():
                correct += 1

        acc = correct / total * 100 if total > 0 else 0
        avg_throughput = total_throughput / total if total > 0 else 0
        summaries.append(
            {
                "Baseline": "MedGraphRAG Ablation",
                "Phase": phase["name"],
                "Correct": correct,
                "Total": total,
                "Accuracy": f"{acc:.2f}%",
                "Throughput": f"{avg_throughput:.2f} tok/s"
            }
        )

    print("\n" + "=" * 80)
    print("  FINAL LOCAL ABLATION SUMMARY (Generator: Qwen 2.5-7B Base)")
    print("=" * 80)
    
    print(f"  {'Baseline':<25} {'Accuracy':<10} {'Score':<10} {'Throughput':<15} {'Phase'}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*15} {'-'*20}")
    for s in summaries:
        score = f"{s['Correct']}/{s['Total']}"
        thr = s.get("Throughput", "N/A")
        print(f"  {s['Baseline']:<25} {s['Accuracy']:<10} {score:<10} {thr:<15} {s['Phase']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
