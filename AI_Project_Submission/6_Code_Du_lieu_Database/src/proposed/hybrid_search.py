import sys
import os

# Ensure the root directory is in sys.path to import retriever
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.proposed.retriever import build_retriever, HybridRetriever


class ProposedHybridRetriever:
    def __init__(self):
        self.retriever = build_retriever()

    def retrieve(self, query: str, top_k: int = 3, mode: str = "hybrid") -> list:
        if mode == "none":
            return []
        elif mode == "bm25":
            return self.retriever._bm25_search(query, top_k=top_k)
        elif mode == "dense":
            return self.retriever._dense_search(query, top_k=top_k)
        else:
            return self.retriever.retrieve(query, final_top_k=top_k, use_rerank=True)
