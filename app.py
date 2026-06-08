"""
app.py — FastAPI server for the Vietnamese Medical Chatbot

Start with:
    uvicorn app:app --host 0.0.0.0 --port 8000

Environment variables (optional):
    MODEL_PATH   — path to merged model dir or LoRA adapter dir
    USE_MERGED   — "1" = merged 16-bit model (default), "0" = LoRA adapters
    NO_RAG       — "1" to disable RAG retrieval
    PORT         — server port (default 8000)
"""

from __future__ import annotations

import os

# ── Must be set before ANY torch/triton import ──────────────────────────────
# Blackwell GPU (sm_120a) is not yet supported by the bundled ptxas.
# Disabling torch.compile and dynamo prevents Triton kernel compilation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"]     = "1"   # disable torch.compile globally
os.environ["TORCH_COMPILE_DISABLE"]   = "1"

import re
import time
import torch
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ───────────────────────────── Config ──────────────────────────────────────
MODEL_PATH     = os.environ.get("MODEL_PATH", "/kaggle/working/models/qwen_chatdoctor_lora_new_dataset")
USE_MERGED     = os.environ.get("USE_MERGED", "1") == "1"   # default: merged model
DISABLE_RAG    = os.environ.get("NO_RAG", "0") == "1"
MAX_SEQ_LEN    = 1536
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "You are an expert medical AI assistant. "
    "You will be provided with medical abstracts as context. "
    "Your task is to carefully read the context and use it to answer the user's question. "
    "Formulate a detailed explanation based on the context, and conclude with a "
    "final decision of Yes, No, or Maybe."
)

# ───────────────────────────── Singletons ──────────────────────────────────
_model     = None
_tokenizer = None
_retriever = None


# ───────────────────────────── Startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _retriever

    # ── Load model with plain HuggingFace (no Unsloth Triton kernels) ──────
    print("🔄 Loading model and tokenizer (HuggingFace backend)...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token     = _tokenizer.eos_token
        _tokenizer.pad_token_id  = _tokenizer.eos_token_id

    if USE_MERGED:
        # Merged 16-bit → load in 4-bit to save VRAM
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # LoRA adapters — load base + adapter
        from peft import AutoPeftModelForCausalLM
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _model = AutoPeftModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )

    _model.eval()
    print("✅ Model loaded.")

    # ── RAG retriever (CPU embeddings, GPU stays free for LLM) ─────────────
    if not DISABLE_RAG:
        print("🔄 Building RAG retriever (CPU embeddings)...")
        try:
            import vector_db as _vdb_mod
            # Monkey-patch VectorDB to force CPU device for the embedding model
            _orig_init = _vdb_mod.VectorDB.__init__
            def _cpu_init(self, db_path="./chroma_db"):
                from sentence_transformers import SentenceTransformer
                import chromadb
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                self.device = "cpu"
                self.model  = SentenceTransformer("NeuML/pubmedbert-base-embeddings", device="cpu")
                self.client = chromadb.PersistentClient(path=db_path)
                self.collection = self.client.get_or_create_collection(
                    name="MedRAG_Hybrid",
                    metadata={"description": "Medical textbooks and PubMed embeddings", "hnsw:space": "cosine"},
                )
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500, chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " ", ""],
                )
            _vdb_mod.VectorDB.__init__ = _cpu_init

            from retriever import build_retriever
            _retriever = build_retriever()
            print("✅ RAG retriever ready (CPU).")
        except Exception as e:
            print(f"⚠️  RAG retriever failed: {e}. Continuing without RAG.")

    yield
    print("👋 Shutting down.")


# ───────────────────────────── App ─────────────────────────────────────────
app = FastAPI(
    title="Vietnamese Medical Chatbot",
    description="LLM-powered medical Q&A with hybrid RAG",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ───────────────────────────── Schemas ─────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    use_rag: Optional[bool] = True

class ChatResponse(BaseModel):
    answer: str
    retrieved_docs: int = 0
    latency_ms: float   = 0.0


# ───────────────────────────── Helpers ─────────────────────────────────────
def retrieve_context(question: str, top_k: int = 3) -> tuple[str, int]:
    if _retriever is None:
        return "", 0
    try:
        docs = _retriever.retrieve(question, final_top_k=top_k, use_rerank=True)
        return "\n\n".join(d["text"] for d in docs), len(docs)
    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return "", 0


def build_prompt(question: str, context: str) -> str:
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}" if context else question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_generation(prompt: str) -> str:
    torch.cuda.empty_cache()

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
    ).to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ───────────────────────────── Routes ──────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return HTMLResponse(content=open(index_path, encoding="utf-8").read())
    return HTMLResponse(content="<h1>index.html not found in ./static/</h1>", status_code=404)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "rag_enabled":  _retriever is not None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    t0 = time.perf_counter()

    context, n_docs = "", 0
    if req.use_rag and _retriever is not None:
        context, n_docs = retrieve_context(req.question)

    prompt  = build_prompt(req.question, context)
    answer  = run_generation(prompt)

    latency_ms = (time.perf_counter() - t0) * 1000
    return ChatResponse(answer=answer, retrieved_docs=n_docs, latency_ms=round(latency_ms, 1))


# ───────────────────────────── Dev runner ──────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
