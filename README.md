# 🩺 Vietnamese Medical Chatbot

An AI-powered medical question-answering system built on **Qwen 2.5-7B** fine-tuned with **QLoRA** on [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA), with a **Hybrid RAG** pipeline (dense + BM25 + cross-encoder reranking) backed by 127K+ medical document chunks from MedRAG textbooks and PubMed.

---

## ✨ Features

- 🤖 **Fine-tuned LLM** — Qwen 2.5-7B-Instruct with LoRA adapters, trained on PubMedQA
- 📚 **Hybrid RAG** — Dense vector search + BM25 + MedCPT cross-encoder reranker
- 🗂️ **127K+ document chunks** — MedRAG textbooks + PubMed abstracts in ChromaDB
- 🌐 **Web Chat UI** — Dark-mode glassmorphism interface with RAG toggle
- ⚡ **CLI inference** — Single-shot or interactive REPL mode
- 🔁 **Resumable training** — Checkpoint resume, early stopping, best/last checkpoint tracking

---

## 📁 Project Structure

```
vietnamese-medical-chatbot/
├── trainer.py              # Fine-tuning script (Unsloth + SFT)
├── ai-project-rag.ipynb    # Kaggle notebook version of trainer.py
├── main.py                 # CLI inference script
├── app.py                  # FastAPI web server
├── static/
│   └── index.html          # Chat UI
├── retriever.py            # Hybrid RAG retriever (BM25 + dense + reranker)
├── vector_db.py            # ChromaDB vector store wrapper
├── benchmark.py            # Evaluation / benchmarking script
└── chroma_db/              # Persistent ChromaDB storage
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Activate your conda environment
conda activate medical_fix

# Install dependencies
pip install fastapi uvicorn peft bitsandbytes transformers \
            sentence-transformers chromadb langchain-text-splitters \
            datasets trl unsloth
```

---

## 🌐 Option 1 — Web Chat UI (Recommended)

Start the FastAPI server and open the browser-based chat interface.

```bash
cd /path/to/vietnamese-medical-chatbot

MODEL_PATH=/path/to/models/qwen_phase1_merged \
USE_MERGED=1 \
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | *(required)* | Path to merged model dir or LoRA adapter dir |
| `USE_MERGED` | `1` | `1` = load merged 16-bit model (recommended); `0` = LoRA adapters |
| `NO_RAG` | `0` | `1` = disable RAG (answer directly, faster) |
| `PORT` | `8000` | Server port |

### Examples

```bash
# Merged model + RAG (default, recommended)
MODEL_PATH=./models/qwen_phase1_merged USE_MERGED=1 uvicorn app:app --host 0.0.0.0 --port 8000

# LoRA adapters (requires base model to be downloaded)
MODEL_PATH=./models/qwen_chatdoctor_lora USE_MERGED=0 uvicorn app:app --host 0.0.0.0 --port 8000

# No RAG (faster, no ChromaDB needed)
MODEL_PATH=./models/qwen_phase1_merged USE_MERGED=1 NO_RAG=1 uvicorn app:app --host 0.0.0.0 --port 8000

# Custom port
MODEL_PATH=./models/qwen_phase1_merged USE_MERGED=1 PORT=9000 python app.py
```

---

## 💻 Option 2 — CLI Inference

Run inference directly from the terminal.

### Single question

```bash
python main.py --question "Does aspirin reduce fever?"
```

### Interactive REPL (keep chatting)

```bash
python main.py
```

### All CLI flags

```bash
python main.py \
  --question "Can statins prevent heart attacks?" \   # omit for interactive mode
  --model_path ./models/qwen_phase1_merged \          # path to model
  --use_merged \                                       # load as merged 16-bit model
  --no_rag \                                           # disable RAG retrieval
  --max_new_tokens 512                                 # max tokens to generate
```

| Flag | Default | Description |
|---|---|---|
| `--question` / `-q` | *(none → interactive)* | Single question to answer |
| `--model_path` | `/kaggle/working/models/...` | Model directory |
| `--use_merged` | off | Load merged 16-bit model instead of LoRA |
| `--no_rag` | off | Disable RAG (no ChromaDB needed) |
| `--max_new_tokens` | `512` | Max tokens to generate |

---

## 🏋️ Option 3 — Training

### On Kaggle (recommended, free T4 GPU)

1. Upload `ai-project-rag.ipynb` to [Kaggle](https://kaggle.com)
2. Enable **GPU T4 x2** accelerator
3. Run all cells

### Locally

```bash
python trainer.py
```

Training will automatically **resume from the last checkpoint** if one exists in `OUTPUT_DIR`.

### Checkpoint layout after training

```
models/qwen_chatdoctor_lora_new_dataset/
├── checkpoint-100/          # step checkpoints (kept: last 3)
├── checkpoint-200/
├── checkpoint-last/         # ← always the most recent eval checkpoint
├── checkpoint-best/         # ← best accuracy checkpoint
├── best_checkpoint_meta.json   # step, accuracy, source path
└── adapter_model.safetensors   # final saved adapter
```

### Key training features

| Feature | Detail |
|---|---|
| **Validation** | `model.generate()` on 200 held-out PubMedQA examples per eval step |
| **Early stopping** | Stops after **5 consecutive eval steps** with no accuracy improvement |
| **Checkpoints** | `checkpoint-last` (always) + `checkpoint-best` (on improvement) |
| **Resume** | Auto-detects last checkpoint and resumes |
| **Metric** | `Conclusion: Yes/No/Maybe` accuracy |

---

## 🔌 API Reference

### `POST /chat`

```json
// Request
{ "question": "Does vitamin D reduce cancer risk?", "use_rag": true }

// Response
{
  "answer": "Based on the provided context...\n\nConclusion: Maybe.",
  "retrieved_docs": 3,
  "latency_ms": 4231.5
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "rag_enabled": true }
```

### `GET /`

Serves the web chat UI.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Fine-tuning | QLoRA (r=32, α=64) on PubMedQA `pqa_artificial` |
| Dataset | ~210K examples, filtered to max 1536 tokens |
| Training | SFT on assistant responses only |
| Inference | HuggingFace `AutoModelForCausalLM` + 4-bit NF4 bitsandbytes |

## 📚 RAG Details

| Property | Value |
|---|---|
| Vector DB | ChromaDB (cosine similarity) |
| Embedding model | `NeuML/pubmedbert-base-embeddings` (CPU) |
| Reranker | `ncbi/MedCPT-Cross-Encoder` |
| Search | Hybrid: dense (60%) + BM25 (40%), RRF merge |
| Corpus | MedRAG textbooks + PubMed (~127K chunks) |

---

## ⚠️ Disclaimer

This chatbot is for **research and educational purposes only**. It is **not a substitute for professional medical advice**, diagnosis, or treatment. Always consult a qualified healthcare provider.