==================================================
THÔNG TIN SOURCE CODE VÀ DATABASE
==================================================

1. Cấu trúc thư mục:
- /src: Chứa toàn bộ mã nguồn của dự án.
  + /proposed: Pipeline chính của nhóm, tích hợp Hybrid RAG (Dense + BM25) kết hợp với mô hình LLM.
    * app.py: Chứa API Server (FastAPI).
    * pipeline.py: Định nghĩa quy trình Retrieval-Augmented Generation.
    * hybrid_search.py, vector_db.py: Module cơ sở dữ liệu vector và tìm kiếm.
  + /baselines: Chứa code để đánh giá các mô hình SOTA hiện nay như BioMistral.
- /static (nếu có): Chứa giao diện frontend (HTML/CSS/JS).
- requirements.txt: Danh sách các thư viện cần thiết.
- README.md: Hướng dẫn kỹ thuật và tóm tắt cấu trúc.

2. Hướng dẫn cài đặt sơ bộ:
- Cài đặt môi trường Python >= 3.10.
- Chạy lệnh: pip install -r requirements.txt
- Đặt biến môi trường MODEL_PATH trỏ đến mô hình đã huấn luyện.
- Chạy: python -m src.proposed.app và truy cập http://localhost:8000.

3. Dữ liệu (Database):
- Dữ liệu tài liệu y khoa được mã hóa dưới dạng Vector (ChromaDB) được khởi tạo ngầm định trong class VectorDB (src/proposed/vector_db.py). Hệ thống tự động sử dụng NeuML/pubmedbert-base-embeddings.
