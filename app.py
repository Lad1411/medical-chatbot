"""
app.py — FastAPI server for the Vietnamese Medical Chatbot

Start with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Environment variables (optional):
    MODEL_PATH   — path to LoRA adapter dir  (default: /kaggle/working/models/qwen_chatdoctor_lora_new_dataset)
    USE_MERGED   — set to "1" to load merged 16-bit model
    NO_RAG       — set to "1" to disable RAG retrieval
    PORT         — override the default port (used by uvicorn in __main__)
"""

import os
import re
import time
import torch
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─────────────────────────── Config ───────────────────────────
MODEL_PATH   = os.environ.get("MODEL_PATH", "/kaggle/working/models/qwen_chatdoctor_lora_new_dataset")
USE_MERGED   = os.environ.get("USE_MERGED", "0") == "1"
DISABLE_RAG  = os.environ.get("NO_RAG", "0") == "1"
MAX_SEQ_LEN  = 1536
MAX_NEW_TOKENS = 512
LOAD_IN_4BIT = not USE_MERGED

SYSTEM_PROMPT = (
    "You are an expert medical AI assistant. "
    "You will be provided with medical abstracts as context. "
    "Your task is to carefully read the context and use it to answer the user's question. "
    "Formulate a detailed explanation based on the context, and conclude with a "
    "final decision of Yes, No, or Maybe."
)

# ─────────────────────────── Global singletons ────────────────
_model     = None
_tokenizer = None
_retriever = None


# ─────────────────────────── Startup / Shutdown ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup."""
    global _model, _tokenizer, _retriever

    print("🔄 Loading model and tokenizer...")
    from unsloth import FastLanguageModel
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(_model)
    _model.eval()
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.pad_token_id = _tokenizer.eos_token_id
    print("✅ Model loaded.")

    if not DISABLE_RAG:
        print("🔄 Building RAG retriever...")
        try:
            from retriever import build_retriever
            _retriever = build_retriever()
            print("✅ RAG retriever ready.")
        except Exception as e:
            print(f"⚠️  RAG retriever failed to load: {e}. Continuing without RAG.")

    yield

    print("👋 Shutting down...")


# ─────────────────────────── App ──────────────────────────────
app = FastAPI(
    title="Vietnamese Medical Chatbot",
    description="LLM-powered medical Q&A with hybrid RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files (index.html lives in ./static/)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─────────────────────────── Schemas ──────────────────────────
class ChatRequest(BaseModel):
    question: str
    use_rag: Optional[bool] = True

class ChatResponse(BaseModel):
    answer: str
    retrieved_docs: int = 0
    latency_ms: float = 0.0


# ─────────────────────────── Helpers ──────────────────────────
def retrieve_context(question: str, top_k: int = 3) -> tuple[str, int]:
    if _retriever is None:
        return "", 0
    try:
        docs = _retriever.retrieve(question, final_top_k=top_k, use_rerank=True)
        context_str = "\n\n".join(d["text"] for d in docs)
        return context_str, len(docs)
    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return "", 0


def build_prompt(question: str, context: str) -> str:
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}" if context else question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_generation(prompt: str) -> str:
    device = next(_model.parameters()).device
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
    ).to(device)

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


# ─────────────────────────── Routes ───────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found in ./static/</h1>", status_code=404)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "rag_enabled": _retriever is not None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    t0 = time.perf_counter()

    # RAG retrieval
    context, n_docs = ("", 0)
    if req.use_rag and _retriever is not None:
        context, n_docs = retrieve_context(req.question)

    # Build prompt & generate
    prompt = build_prompt(req.question, context)
    answer = run_generation(prompt)

    latency_ms = (time.perf_counter() - t0) * 1000

    return ChatResponse(
        answer=answer,
        retrieved_docs=n_docs,
        latency_ms=round(latency_ms, 1),
    )


# ─────────────────────────── Dev runner ───────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
