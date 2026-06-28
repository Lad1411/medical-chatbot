import sys
import os
import argparse
from typing import List, Dict
from tqdm import tqdm
from datasets import load_dataset
import re
from collections import defaultdict

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.proposed.llm import BaseGenerator
from src.proposed.retriever import build_retriever

class MedMetaGraphConstructor:
    def __init__(self):
        self.graph = defaultdict(set)
        self.entity_to_doc = defaultdict(list)

    def extract_entities(self, text: str) -> List[str]:
        """
        Robust heuristic entity extraction for Medical texts.
        Extracts Capitalized phrases, acronyms, and common medical suffixes.
        """
        # Match Acronyms (e.g. COPD, HIV) or Capitalized medical terms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        title_cased = re.findall(r'\b[A-Z][a-z]{3,}(?:\s+[a-z]{3,})?\b', text)
        
        entities = list(set(acronyms + title_cased))
        return entities

    def build_local_graph(self, documents: List[Dict]):
        """Builds a local knowledge graph where nodes are entities and edges are co-occurrences.
           Also maps entities back to their source documents."""
        self.graph.clear()
        self.entity_to_doc.clear()
        
        for doc in documents:
            text = doc["text"]
            entities = self.extract_entities(text)
            for i in range(len(entities)):
                self.entity_to_doc[entities[i]].append(doc)
                for j in range(i + 1, min(i + 5, len(entities))):  # Link close entities (window=5)
                    self.graph[entities[i]].add(entities[j])
                    self.graph[entities[j]].add(entities[i])


class GraphTraversalRetriever:
    def __init__(self, graph_constructor, base_retriever):
        self.graph_constructor = graph_constructor
        self.base_retriever = base_retriever

    def retrieve_subgraph_context(self, query: str, mode: str = "full") -> List[Dict]:
        # Step 1: Ablation thresholds for Graph retrieval
        if mode == "baseline":
            top_k, max_nodes = 5, 5
        elif mode == "metagraph":
            top_k, max_nodes = 10, 10
        elif mode == "triplegraph":
            top_k, max_nodes = 15, 20
        else:  # full
            top_k, max_nodes = 20, 30

        # Step 2: Retrieve local broad context (Initial dense retrieval)
        docs = self.base_retriever._dense_search(query, top_k=top_k)
        
        # Step 3: Build local graph on-the-fly
        self.graph_constructor.build_local_graph(docs)

        # Step 4: Extract query entities and traverse graph
        query_entities = set(self.graph_constructor.extract_entities(query))
        
        context_nodes = set()
        for q_ent in query_entities:
            if q_ent in self.graph_constructor.graph:
                context_nodes.add(q_ent)
                # 1-hop traversal
                context_nodes.update(self.graph_constructor.graph[q_ent])
                
        # If no query entities match, fallback to highest degree nodes (PageRank proxy)
        if not context_nodes and self.graph_constructor.graph:
            sorted_nodes = sorted(
                self.graph_constructor.graph.keys(), 
                key=lambda k: len(self.graph_constructor.graph[k]), 
                reverse=True
            )
            context_nodes.update(sorted_nodes[:max_nodes])

        context_nodes = list(context_nodes)[:max_nodes]
        
        # Step 5: Map subgraph nodes back to actual Document Text chunks
        retrieved_docs = []
        seen_docs = set()
        for node in context_nodes:
            for doc in self.graph_constructor.entity_to_doc.get(node, []):
                doc_id = doc.get("id", doc["text"][:50])
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    retrieved_docs.append(doc)
                    
        # Limit to top 3 most relevant documents associated with the subgraph to fit context window
        return retrieved_docs[:3]


class MedGraphRAGPipeline:
    def __init__(self, lora_path: str = None, mode: str = "full"):
        self.mode = mode
        self.graph_constructor = MedMetaGraphConstructor()
        self.base_retriever = build_retriever()
        self.retriever = GraphTraversalRetriever(self.graph_constructor, self.base_retriever)
        self.generator = BaseGenerator(lora_path=lora_path, is_unsloth=True)

    def run(self, query: str, options: dict = None, dataset_type: str = "medqa") -> tuple:
        if dataset_type == "medqa":
            options_text = "\n".join([f"{k}) {v}" for k, v in (options or {}).items()])
            full_query = f"{query}\n{options_text}"
        else:
            full_query = query

        context_docs = self.retriever.retrieve_subgraph_context(full_query, mode=self.mode)
        
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
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    print("[+] Initializing Genuine Local MedGraphRAG Pipeline")
    pipeline = MedGraphRAGPipeline()
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    correct = 0
    total = len(dataset)
    total_throughput = 0.0
    for row in tqdm(dataset, desc="MedQA MedGraphRAG"):
        pred_text, thr = pipeline.run(
            row["question"], options=row["options"], dataset_type="medqa"
        )
        pred = extract_mcq_answer(pred_text)
        total_throughput += thr
        if str(pred).upper() == str(row["answer_idx"]).upper():
            correct += 1

    acc = correct / total * 100 if total > 0 else 0
    avg_throughput = total_throughput / total if total > 0 else 0
    print(f"\n[+] MedGraphRAG Accuracy: {acc:.2f}% ({correct}/{total}) | Throughput: {avg_throughput:.2f} tok/s")


if __name__ == "__main__":
    main()
