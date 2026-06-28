# 🩺 Vietnamese Medical Chatbot

An AI-powered medical question-answering system built on **Qwen 2.5-7B** fine-tuned with **QLoRA** on [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA), with a **Hybrid RAG** pipeline (dense + BM25 + cross-encoder reranking) backed by 2.3 million+ medical document chunks from MedRAG textbooks and PubMed.

---

## ✨ Features

- 🤖 **Fine-tuned LLM** — Qwen 2.5-7B-Instruct with LoRA adapters, trained on PubMedQA
- 📚 **Hybrid RAG** — Dense vector search + BM25 + MedCPT cross-encoder reranker
- 🗂️ **2.3 million+ document chunks** — MedRAG textbooks + PubMed abstracts in ChromaDB
- 🌐 **Web Chat UI** — Dark-mode glassmorphism interface with RAG toggle
- ⚡ **CLI inference** — Single-shot or interactive REPL mode
- 📊 **Robust Evaluation Suite** — Run baseline ablations (MedRAG, MedGraphRAG) and SOTA local models (Cure-Med 14B/32B).

---

## 📁 Project Structure

```
vietnamese-medical-chatbot/
├── README.md
├── requirements.txt
├── docs/                      # Documentation, LaTeX report, and defense slides
├── static/
│   └── index.html             # Web Chat UI
└── src/
    ├── proposed/              # Proposed Qwen 7B + QLoRA + Hybrid RAG system
    │   ├── app.py             # FastAPI web server
    │   ├── main.py            # CLI inference script
    │   ├── trainer.py         # QLoRA fine-tuning script
    │   ├── run_benchmark.py   # Benchmark script for the proposed architecture
    │   ├── pipeline.py        # End-to-End RAG + LLM pipeline wrapper
    │   ├── retriever.py       # Hybrid RAG retriever implementation
    │   ├── vector_db.py       # ChromaDB vector store wrapper
    │   └── llm.py             # Base generator logic
    └── baselines/             # Ablation and local SOTA baselines
        ├── run_all_baselines.py     # Executes MedRAG/MedGraphRAG ablation studies
        ├── medrag_pipeline.py       # Local MedRAG pipeline simulator
        ├── medgraphrag_pipeline.py  # Local MedGraphRAG pipeline simulator
        └── cure_med_pipeline.py     # Local runner for Cure-Med 14B and 32B
```

---

## 🚀 Quick Start

### 1. Prerequisites

Create a new Conda environment and install dependencies:

```bash
# Activate your conda environment
conda create -n medical_chatbot python=3.10 -y
conda activate medical_chatbot

# Install dependencies
pip install -r requirements.txt
```

*(Ensure you have NVIDIA drivers and PyTorch with CUDA support installed).*

### 2. Environment Variables
You may need to export environment variables if you are using specific models or APIs:
```bash
export OPENAI_API_KEY="your_key" # If evaluating API baselines
```

---

## 🌐 Running the Web Application (Recommended)

Start the FastAPI server and open the browser-based chat interface. The server code is located inside `src/proposed/app.py`.

```bash
# From the project root directory
cd src/proposed/

# Run the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser to interact with the dark-mode glassmorphism UI.

### App Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Path to the base foundation model |
| `LORA_PATH` | *(none)* | Path to trained LoRA adapters |
| `NO_RAG` | `0` | Set to `1` to disable Hybrid RAG retrieval (answers directly) |
| `PORT` | `8000` | Web server port |

*Example running with custom LoRA weights and RAG:*
```bash
LORA_PATH=/path/to/your/lora_checkpoints uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 💻 Running the CLI (Command Line Interface)

Run the inference script directly from the terminal.

### Interactive Chat Mode
```bash
cd src/proposed/
python main.py
```

### Single Question Mode
```bash
cd src/proposed/
python main.py --question "Does aspirin reduce fever?"
```

### All CLI flags for `main.py`
| Flag | Description |
|---|---|
| `--question` | Single question to answer (omit for interactive chat) |
| `--model_path` | Base model directory/HF hub name |
| `--lora_path` | Trained LoRA checkpoint path |
| `--no_rag` | Disable Hybrid RAG retrieval |
| `--max_new_tokens` | Max tokens to generate (default: 512) |

---

## 📊 Running Evaluations & Baselines

All evaluations are split into evaluating the **Proposed Architecture** and evaluating the **Local Baselines** (Ablations).

### 1. Evaluate the Proposed Architecture
To evaluate the `Qwen 2.5-7B + QLoRA + Hybrid RAG` system on MedQA:
```bash
cd src/proposed/
python run_benchmark.py --dataset medqa --mode proposed
```

*(Use `--limit 100` to run a smaller subset for quick testing).*

### 2. Run MedRAG & MedGraphRAG Ablation Studies (Table 2)
To reproduce the structural ablation studies isolating the effect of RAG components (BM25, Dense, Hybrid) without QLoRA:
```bash
cd src/baselines/
python run_all_baselines.py
```
*Note: This script strictly runs local ablations using the Qwen 2.5-7B zero-shot backbone. SOTA models like GPT-4, LLaMA-2 70B, and CURE are cited from literature.*

### 3. Run Cure-Med 14B / 32B Baselines (Table 1)
To benchmark the local state-of-the-art open-weights models (Cure-Med):
```bash
cd src/baselines/
# Run 14B model
python cure_med_pipeline.py --model 14b

# Run 32B model (Requires CPU Offloading / device_map="auto")
python cure_med_pipeline.py --model 32b
```

---

## 🏋️ Training (QLoRA Fine-Tuning)

To fine-tune the Qwen 2.5-7B model using QLoRA on the artificial PubMedQA dataset:

```bash
cd src/proposed/
python trainer.py
```

Training will automatically **resume from the last checkpoint** if one exists.
Checkpoints are saved locally. Early stopping is implemented to halt training if validation accuracy plateaus for 5 consecutive evaluation steps.

---

## ⚠️ Disclaimer

This chatbot and its associated models are for **research and educational purposes only**. They are **not a substitute for professional medical advice**, diagnosis, or treatment. Always consult a qualified healthcare provider.