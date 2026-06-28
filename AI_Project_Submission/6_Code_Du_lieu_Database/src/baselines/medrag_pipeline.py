import sys
import os
import argparse
import csv
from tqdm import tqdm
from datasets import load_dataset
import re

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.proposed.retriever import build_retriever
from src.proposed.llm import BaseGenerator


class MedRAGPipeline:
    def __init__(self, retriever_type: str = "bm25", lora_path: str = None):
        print(f"[+] Initializing MedRAG Pipeline (Mode: {retriever_type})")
        self.retriever_type = retriever_type
        self.retriever = build_retriever()
        self.generator = BaseGenerator(lora_path=lora_path, is_unsloth=True)

    def run(self, query: str, options: dict = None, dataset_type: str = "medqa") -> tuple:
        if dataset_type == "medqa":
            options_text = "\n".join([f"{k}) {v}" for k, v in (options or {}).items()])
            full_query = f"{query}\n{options_text}"
        else:
            full_query = query

        if self.retriever_type == "none":
            context_docs = []
        elif self.retriever_type == "bm25":
            context_docs = self.retriever._bm25_search(full_query, top_k=3)
        elif self.retriever_type == "dense":
            context_docs = self.retriever._dense_search(full_query, top_k=3)
        else:
            context_docs = self.retriever.retrieve(
                full_query, final_top_k=3, use_rerank=False
            )

        pred_text, throughput = self.generator.generate(
            context_docs, query, options, dataset_type=dataset_type
        )
        return pred_text, throughput


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="bm25",
        choices=["bm25", "dense", "hybrid"],
        help="Retriever mode",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pipeline = MedRAGPipeline(retriever_type=args.mode)
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    correct = 0
    total = len(dataset)
    total_throughput = 0.0
    for row in tqdm(dataset, desc=f"MedQA MedRAG ({args.mode})"):
        pred_text, thr = pipeline.run(
            row["question"], options=row["options"], dataset_type="medqa"
        )
        pred = extract_mcq_answer(pred_text)
        total_throughput += thr
        if str(pred).upper() == str(row["answer_idx"]).upper():
            correct += 1

    acc = correct / total * 100 if total > 0 else 0
    avg_throughput = total_throughput / total if total > 0 else 0
    print(f"\n[+] MedRAG {args.mode.upper()} Accuracy: {acc:.2f}% ({correct}/{total}) | Throughput: {avg_throughput:.2f} tok/s")


if __name__ == "__main__":
    main()
