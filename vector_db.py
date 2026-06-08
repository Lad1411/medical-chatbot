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

    def process_and_upsert_dataset(self, repo_id: str, max_samples: int = None, chunk_batch_size: int = 512):
        print(f"-> Đang tải & upsert {repo_id}...")
        dataset = load_dataset(repo_id, split="train", streaming=True)
        
        if max_samples:
            dataset = dataset.take(max_samples)

        chunk_buffer = []
        pbar = tqdm(desc=f"Xử lý {repo_id}", total=max_samples)
        
        for row in dataset:
            text = row['contents']
            title = row.get('title', 'Unknown Title')
            doc_id = str(row['id'])

            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_buffer.append({
                    "id": f"{repo_id}_{doc_id}_chunk_{i}",
                    "text": chunk,
                    "metadata": {
                        "title": title,
                        "source": repo_id,         
                        "parent_id": doc_id,
                        "chunk_index": i
                    }
                })
                
                # Embed and upsert when buffer reaches batch size
                if len(chunk_buffer) >= chunk_batch_size:
                    self._upsert_batch(chunk_buffer)
                    chunk_buffer = []
            
            pbar.update(1)
            
        # Process any remaining chunks
        if chunk_buffer:
            self._upsert_batch(chunk_buffer)
            
        pbar.close()

    def _upsert_batch(self, batch: list):
        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        embeddings = self.model.encode(
            texts,
            batch_size=len(texts),
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

    def build_db(self, batch_size: int = 512, pubmed_max_samples: int = None, textbooks_max_samples: int = None):
        print("🔨 Bắt đầu build VectorDB (Textbooks + PubMed)...")
        # Textbooks dataset is smaller, but we stream-upsert it as well for consistency
        self.process_and_upsert_dataset("MedRAG/textbooks", max_samples=textbooks_max_samples, chunk_batch_size=batch_size)
        self.process_and_upsert_dataset("MedRAG/pubmed", max_samples=pubmed_max_samples, chunk_batch_size=batch_size)
        print("✅ Hoàn thành build VectorDB!")


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