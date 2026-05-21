import os
from sentence_transformers import SentenceTransformer
import chromadb
from datasets import load_dataset
from tqdm import tqdm


class VectorDB:
    def __init__(self, db_path: str = "./chroma_db"):
        self.device = "gpu"
        # print(f"Đang chạy trên thiết bị: {self.device}")

        self.model = SentenceTransformer("NeuML/pubmedbert-base-embeddings", device=self.device)
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name="MedRAG",
            metadata={
                "description": "Medical textbook embeddings",
                "hnsw:space": "cosine"
            }
        )

    def load_dataset(self):
        # Tối ưu: Đưa biến giới hạn data ra ngoài để linh hoạt mở rộng quy mô sau này
        dataset = load_dataset(
            "MedRAG/textbooks",
            split="train"
        )
        return dataset

    def embedding_texts(self, texts, batch_size: int = 32):
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def build_db(self, batch_size: int = 32):
        dataset = self.load_dataset()
        total = len(dataset)

        for i in tqdm(range(0, total, batch_size)):
            batch = dataset[i: i + batch_size]
            texts = batch['contents']

            # Đồng bộ hóa batch_size cho tầng embedding xử lý trên CPU
            embeddings = self.embedding_texts(texts, batch_size=batch_size)
            ids = [str(x) for x in batch['id']]

            metadatas = []
            for j in range(len(texts)):
                metadata = {
                    "title": batch["title"][j],
                    "source": "MedRAG/textbooks",
                    "chunk_id": batch["id"][j]
                }
                metadatas.append(metadata)

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


if __name__ == "__main__":
    db = VectorDB()

    # Kiểm tra trạng thái DB
    existing_count = db.collection.count()
    print(f"Số lượng tài liệu hiện có trong Database: {existing_count}")

    if existing_count == 0:
        print("-> Database trống! Đang tiến hành tạo dữ liệu...")
        # Sử dụng batch_size=32 là con số tối ưu cho CPU (không quá nặng, không quá lắt nhắt)
        db.build_db(limit=100, batch_size=32)
        print("-> Đã nạp dữ liệu xong!")
    else:
        print("-> Dữ liệu đã tồn tại sẵn trên ổ đĩa. Sẵn sàng truy vấn.")

    print("=" * 50)
    print("Đang tiến hành tìm kiếm...")

    query_text = "What causes heart failure?"
    results = db.search(query_text, top_k=3)

    # Đọc kết quả mượt mà và tường minh hơn
    for i in range(len(results["documents"][0])):
        print("=" * 50)
        print(f"Top {i + 1} Match - Title: {results['metadatas'][0][i]['title']}")
        print(f"Nội dung: {results['documents'][0][i][:500]}...")
        print(f"Distance (Cosine): {results['distances'][0][i]:.4f}")