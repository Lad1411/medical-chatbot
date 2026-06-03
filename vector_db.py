import os
from sentence_transformers import SentenceTransformer
import chromadb
from datasets import load_dataset
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch

class VectorDB:
    def __init__(self, db_path: str = "./chroma_db"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = SentenceTransformer("NeuML/pubmedbert-base-embeddings", device=self.device)
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name="MedRAG_Hybrid",
            metadata={
                "description": "Medical textbooks and PubMed embeddings",
                "hnsw:space": "cosine"
            }
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  
            chunk_overlap=200, 
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_dataset(self, repo_id: str, max_samples: int = None):
        print(f"-> Đang tải {repo_id}...")
        dataset = load_dataset(repo_id, split="train", streaming=True)
        
        if max_samples:
            dataset = dataset.take(max_samples)

        chunks_data = []
        for row in tqdm(dataset, desc=f"Cắt chunk {repo_id}", total=max_samples):
            text = row['contents']
            title = row.get('title', 'Unknown Title')
            doc_id = str(row['id'])

            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunks_data.append({
                    "id": f"{repo_id}_{doc_id}_chunk_{i}",
                    "text": chunk,
                    "metadata": {
                        "title": title,
                        "source": repo_id,         
                        "parent_id": doc_id,
                        "chunk_index": i
                    }
                })
        return chunks_data

    def build_db(self, batch_size: int = 32):
        all_chunks = []
        all_chunks.extend(self.process_dataset("MedRAG/textbooks"))
        all_chunks.extend(self.process_dataset("MedRAG/pubmed", max_samples = 100000))
        
        total_chunks = len(all_chunks)
        print(f"\n=> Tổng số chunks cần embedding: {total_chunks}")

        for i in tqdm(range(0, total_chunks, batch_size), desc="Đang Embedding & Upsert"):
            batch = all_chunks[i : i + batch_size]
            
            texts = [item["text"] for item in batch]
            ids = [item["id"] for item in batch]
            metadatas = [item["metadata"] for item in batch]

            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            self.collection.upsert(
                documents=texts,
                embeddings=embeddings.tolist(),
                ids=ids,
                metadatas=metadatas
            )

    def search(self, query, top_k=3):
        query_embeddings = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results