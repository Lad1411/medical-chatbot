import sys
import os
import argparse
import csv
import re
import torch
from tqdm import tqdm
from datasets import load_dataset
import gc

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.proposed.pipeline import ProposedHybridRAGPipeline


def extract_mcq_answer(text: str):
    match = re.search(
        r"(?:Conclusion:|Answer:|answer is|correct (?:answer|option|choice) is)\s*([A-D])",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    match_end = re.search(r"\b([A-D])\b[\.\s]*(?:<\|im_end\|>)?$", text, re.IGNORECASE)
    if match_end:
        return match_end.group(1).upper()
    return None


def extract_yes_no_maybe(text: str):
    match = re.search(r"Conclusion:\s*(Yes|No|Maybe)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    match_end = re.search(
        r"\b(yes|no|maybe)\b[\.\s]*(?:<\|im_end\|>)?$", text, re.IGNORECASE
    )
    if match_end:
        return match_end.group(1).lower()

    return "maybe"


def run_phase(pipeline, dataset, dataset_type, name, mode):
    print(f"\n[*] Running {name}...")
    correct = 0
    total = len(dataset)
    total_throughput = 0.0
    for i, row in enumerate(tqdm(dataset, desc=name)):
        if dataset_type == "medqa":
            pred_text, throughput = pipeline.run(
                row["question"],
                options=row["options"],
                dataset_type="medqa",
                retriever_mode=mode,
            )
            pred = extract_mcq_answer(pred_text)
            total_throughput += throughput
            if str(pred).upper() == str(row["answer_idx"]).upper():
                correct += 1
        elif dataset_type == "pubmedqa":
            context = " ".join(row["CONTEXTS"]) if "CONTEXTS" in row else None
            pred_text, throughput = pipeline.run(
                row["QUESTION"],
                dataset_type="pubmedqa",
                context=context,
                retriever_mode=mode,
            )
            total_throughput += throughput
            pred = extract_yes_no_maybe(pred_text)
            if str(pred).lower() == str(row["final_decision"]).lower():
                correct += 1

    acc = correct / total * 100 if total > 0 else 0
    avg_throughput = total_throughput / total if total > 0 else 0
    print(f"[+] {name} Accuracy: {acc:.2f}% ({correct}/{total}) | Throughput: {avg_throughput:.2f} tok/s")
    return correct, total, acc, avg_throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=str, default="checkpoints/qlora-pubmedqa")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    print("[+] Loading Datasets...")
    medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    pubmedqa_ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    if args.limit > 0:
        medqa_ds = medqa_ds.select(range(min(args.limit, len(medqa_ds))))
        pubmedqa_ds = pubmedqa_ds.select(range(min(args.limit, len(pubmedqa_ds))))

    summaries = []

    # ---------------------------------------------------------
    # PART 1: ZERO-SHOT (NO LORA)
    # ---------------------------------------------------------
    print("\n[+] Initializing Base Pipeline (Zero-Shot, No LoRA)...")
    base_pipeline = ProposedHybridRAGPipeline(lora_path=None)

    c, t, a, thr = run_phase(
        base_pipeline,
        medqa_ds,
        "medqa",
        "Qwen 2.5-Instruct (Zero-shot) - No RAG [MedQA]",
        "none",
    )
    summaries.append(("MedQA", "Zero-shot", "none", c, t, a, thr))

    c, t, a, thr = run_phase(
        base_pipeline,
        pubmedqa_ds,
        "pubmedqa",
        "Qwen 2.5-Instruct (Zero-shot) - No RAG [PubMedQA]",
        "none",
    )
    summaries.append(("PubMedQA", "Zero-shot", "none", c, t, a, thr))

    # Free memory
    del base_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PART 2: QLoRA + Ablations
    # ---------------------------------------------------------
    if args.lora and os.path.exists(args.lora):
        print(f"\n[+] Initializing QLoRA Pipeline (Adapter: {args.lora})...")
        qlora_pipeline = ProposedHybridRAGPipeline(lora_path=args.lora)

        ablations = [
            {"name_suffix": "No RAG", "mode": "none"},
            {"name_suffix": "BM25 Only", "mode": "bm25"},
            {"name_suffix": "Dense Only", "mode": "dense"},
            {"name_suffix": "Hybrid RAG", "mode": "hybrid"},
        ]

        for abl in ablations:
            c, t, a, thr = run_phase(
                qlora_pipeline,
                medqa_ds,
                "medqa",
                f"Qwen 2.5 + QLoRA + {abl['name_suffix']} [MedQA]",
                abl["mode"],
            )
            summaries.append(("MedQA", "QLoRA", abl["mode"], c, t, a, thr))

            c, t, a, thr = run_phase(
                qlora_pipeline,
                pubmedqa_ds,
                "pubmedqa",
                f"Qwen 2.5 + QLoRA + {abl['name_suffix']} [PubMedQA]",
                abl["mode"],
            )
            summaries.append(("PubMedQA", "QLoRA", abl["mode"], c, t, a, thr))
    else:
        print(f"\n[!] LoRA path {args.lora} not found. Skipping QLoRA phases.")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("  FINAL PROPOSED ARCHITECTURE SUMMARY (TABLE 2 & 3)")
    print("=" * 80)
    print(
        f"  {'Dataset':<10} {'Model':<12} {'Retriever':<10} {'Accuracy':<10} {'Score':<10} {'Throughput'}"
    )
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for s in summaries:
        score = f"{s[3]}/{s[4]}"
        print(f"  {s[0]:<10} {s[1]:<12} {s[2]:<10} {s[5]:.2f}%      {score:<10} {s[6]:.2f} tok/s")
    print("=" * 80)


if __name__ == "__main__":
    main()
