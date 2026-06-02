"""
retriever.py — Vị trí 2: Advanced RAG Developer (Reranking & Hybrid Search)
Đã được đồng bộ và tối ưu hóa với VectorDB mới.
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

# [CHỈNH SỬA 1]: Import class VectorDB từ file của bạn 
from vector_db import VectorDB 

# ─────────────────────────── Cấu hình ───────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cross-Encoder model tốt cho đa ngôn ngữ + tiếng Việt
RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"

# Số lượng tài liệu lấy ra ban đầu trước khi rerank
INITIAL_DENSE_TOP_K = 20
INITIAL_BM25_TOP_K  = 20

# Số tài liệu sau rerank gửi cho LLM
FINAL_TOP_K = 5

# Trọng số RRF / hybrid
ALPHA_DENSE = 0.6   # tỷ trọng điểm vector search
ALPHA_BM25  = 0.4   # tỷ trọng điểm BM25


# ─────────────────────────── BM25 Index ──────────────────────────
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus: List[str] = []
        self.tokenized: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def fit(self, corpus: List[str]) -> None:
        self.corpus = corpus
        self.N = len(corpus)
        self.tokenized = [self._tokenize(doc) for doc in corpus]
        self.doc_freqs = [Counter(tokens) for tokens in self.tokenized]

        lengths = [len(t) for t in self.tokenized]
        self.avgdl = sum(lengths) / self.N if self.N else 1.0

        df: Counter = Counter()
        for tf in self.doc_freqs:
            for term in tf:
                df[term] += 1

        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.N - freq + 0.5) / (freq + 0.5) + 1
            )

    def get_scores(self, query: str) -> np.ndarray:
        query_terms = self._tokenize(query)
        scores = np.zeros(self.N)

        for term in query_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf in enumerate(self.doc_freqs):
                freq = tf.get(term, 0)
                dl = len(self.tokenized[i])
                numerator   = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i]  += idf * numerator / denominator

        return scores

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ─────────────────────────── Hybrid Retriever ────────────────────
class HybridRetriever:
    def __init__(
        self,
        vector_db: VectorDB, # [CHỈNH SỬA 2]: Đổi type hint thành VectorDB của bạn
        reranker_model: str = RERANKER_MODEL,
    ):
        self.vector_db = vector_db
        self._bm25: Optional[BM25Index] = None
        self._bm25_corpus: List[Dict[str, Any]] = []   
        self._reranker: Optional[CrossEncoder] = None
        self._reranker_model = reranker_model

    def build_bm25_index(self) -> None:
        logger.info("🔨 Đang build BM25 index từ ChromaDB ...")
        
        coll = self.vector_db.collection 

        total = coll.count()
        if total == 0:
            logger.warning("⚠️  ChromaDB rỗng — BM25 index không có dữ liệu.")
            return

        FETCH_BATCH = 5000
        all_docs: List[Dict[str, Any]] = []
        offset = 0

        while offset < total:
            result = coll.get(
                limit=FETCH_BATCH,
                offset=offset,
                include=["documents", "metadatas"],
            )
            for doc_id, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            ):
                all_docs.append({"id": doc_id, "text": text, "metadata": meta})
            offset += FETCH_BATCH

        self._bm25_corpus = all_docs
        corpus_texts = [d["text"] for d in all_docs]

        self._bm25 = BM25Index()
        self._bm25.fit(corpus_texts)
        logger.info(f"✅ BM25 index sẵn sàng: {len(all_docs)} documents.")

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            logger.info(f"🔄 Đang tải Cross-Encoder: {self._reranker_model} ...")
            self._reranker = CrossEncoder(self._reranker_model, max_length=512)
            logger.info("✅ Cross-Encoder sẵn sàng.")
        return self._reranker

    # @staticmethod
    # def _normalize(scores: List[float]) -> List[float]:
    #     if not scores:
    #         return scores
    #     min_s, max_s = min(scores), max(scores)
    #     if max_s == min_s:
    #         return [1.0] * len(scores)
    #     return [(s - min_s) / (max_s - min_s) for s in scores]

    # ── Dense Search ──────────────────────────────────────────────
    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        # [CHỈNH SỬA 4]: Khớp nối hàm search() của bạn và chuẩn hóa định dạng trả về
        raw_results = self.vector_db.search(query, top_k=top_k)
        
        results = []
        if not raw_results["documents"] or not raw_results["documents"][0]:
            return results
            
        for i in range(len(raw_results["documents"][0])):
            # QUAN TRỌNG: VectorDB trả về Khoảng cách (Cosine Distance - Càng THẤP càng tốt)
            # Nhưng thuật toán Hybrid ở dưới lại dùng Score (Càng CAO càng tốt) để tính Normalize & Weight
            # Do đó, ta phải chuyển: Similarity Score = 1.0 - Cosine Distance
            distance = raw_results["distances"][0][i]
            similarity_score = 1.0 - distance 
            
            results.append({
                "id": raw_results["ids"][0][i], # ChromaDB mặc định luôn trả về ids
                "text": raw_results["documents"][0][i],
                "metadata": raw_results["metadatas"][0][i],
                "score": similarity_score # Truyền điểm Similarity vào để tí nữa merge
            })
        return results

    # ── BM25 Search ───────────────────────────────────────────────
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self._bm25 is None or not self._bm25_corpus:
            logger.warning("⚠️  BM25 chưa được build. Gọi build_bm25_index() trước.")
            return []

        ranked = self._bm25.search(query, top_k=top_k)
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            doc = self._bm25_corpus[idx]
            results.append(
                {
                    "id":       doc["id"],
                    "text":     doc["text"],
                    "metadata": doc["metadata"],
                    "score":    score,
                }
            )
        return results

    # ── Hybrid Merge (RRF + weighted score) ───────────────────────
    # ── Hybrid Merge (True RRF + weighted score) ───────────────────────
    def _hybrid_merge(
        self,
        dense_hits: List[Dict[str, Any]],
        bm25_hits:  List[Dict[str, Any]],
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combines Dense and BM25 results using true Reciprocal Rank Fusion (RRF).
        Formula: score = weight * (1 / (rrf_k + rank))
        """
        merged: Dict[str, Dict[str, Any]] = {}

        # 1. Process Dense Hits (List is assumed to be sorted by best score)
        for rank, hit in enumerate(dense_hits, start=1):
            doc_id = hit["id"]
            # Calculate RRF score based purely on rank position
            rrf_score = 1.0 / (rrf_k + rank)
            weighted_score = ALPHA_DENSE * rrf_score
            
            # Initialize the document in our merged dictionary
            merged[doc_id] = {**hit, "hybrid_score": weighted_score}

        # 2. Process BM25 Hits (List is assumed to be sorted by best score)
        for rank, hit in enumerate(bm25_hits, start=1):
            doc_id = hit["id"]
            rrf_score = 1.0 / (rrf_k + rank)
            weighted_score = ALPHA_BM25 * rrf_score
            
            # If the document was also found by Dense search, add to its existing score
            if doc_id in merged:
                merged[doc_id]["hybrid_score"] += weighted_score
            # If it's a new document found only by BM25, add it to the dictionary
            else:
                merged[doc_id] = {**hit, "hybrid_score": weighted_score}

        # 3. Sort final results by the combined RRF hybrid score
        sorted_hits = sorted(
            merged.values(), key=lambda x: x["hybrid_score"], reverse=True
        )
        
        return sorted_hits

    # ── Rerank bằng Cross-Encoder ─────────────────────────────────
    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = FINAL_TOP_K,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        reranker = self._get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = reranker.predict(pairs, show_progress_bar=False)

        for candidate, ce_score in zip(candidates, ce_scores):
            candidate["rerank_score"] = float(ce_score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    # ── Public API ────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        final_top_k: int = FINAL_TOP_K,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        logger.info(f"🔍 Hybrid Retrieval cho query: '{query[:80]}...'")

        dense_hits = self._dense_search(query, top_k=INITIAL_DENSE_TOP_K)
        logger.info(f"   Dense hits: {len(dense_hits)}")

        bm25_hits = self._bm25_search(query, top_k=INITIAL_BM25_TOP_K)
        logger.info(f"   BM25 hits:  {len(bm25_hits)}")

        merged = self._hybrid_merge(dense_hits, bm25_hits)
        logger.info(f"   Sau merge: {len(merged)} candidates")

        if use_rerank and merged:
            final = self._rerank(query, merged, top_k=final_top_k)
            logger.info(f"   Sau rerank: {len(final)} kết quả cuối")
        else:
            final = merged[:final_top_k]

        return final


# ─────────────────────────── Factory ─────────────────────────────
def build_retriever() -> HybridRetriever:
    # [CHỈNH SỬA 5]: Tích hợp cơ chế Smart Load (Chỉ nạp Data khi DB rỗng) vào thẳng Factory
    db = VectorDB()
    
    existing_count = db.collection.count()
    if existing_count == 0:
        logger.info("-> Database Chroma rỗng! Đang tiến hành tạo dữ liệu VectorDB (Chỉ chạy 1 lần)...")
        db.build_db(batch_size=32)
    else:
        logger.info(f"-> Dữ liệu VectorDB đã tồn tại ({existing_count} chunks). Bỏ qua tạo mới.")

    retriever = HybridRetriever(db)
    retriever.build_bm25_index()
    return retriever


# ─────────────────────────── Main ────────────────────────────────
if __name__ == "__main__":
    print("=== Test Hybrid Retriever ===")
    
    # Chỉ cần gọi 1 hàm duy nhất, hệ thống sẽ tự động lo việc DB có cần load hay không
    retriever = build_retriever()

    test_queries = [
        "What is the function of the sacrum and coccyx?",
        "How is extradural anesthesia performed?",
        "What substances are used to attenuate X-rays to demonstrate specific structures?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        hits = retriever.retrieve(q, final_top_k=3, use_rerank=True)
        print(f"Số kết quả: {len(hits)}")
        for i, h in enumerate(hits, 1):
            print(f"\n  [{i}] rerank={h.get('rerank_score', 0):.4f} | hybrid={h.get('hybrid_score', 0):.4f}")
            print(f"      {h['text'][:200]}...")