#!/usr/bin/env python3
"""
main.py — Vietnamese Medical Chatbot Inference CLI

Usage:
    python main.py --question "Does ibuprofen reduce inflammation?"
    python main.py --question "..." --model_path /path/to/lora_adapters --no_rag
    python main.py --question "..." --model_path /path/to/merged_model --use_merged
"""

import argparse
import sys
import torch
from unsloth import FastLanguageModel

# ==========================================
# CONFIGURATION (edit defaults as needed)
# ==========================================
DEFAULT_MODEL_PATH = "/kaggle/working/models/qwen_chatdoctor_lora_new_dataset"
BASE_MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1536
MAX_NEW_TOKENS = 512
LOAD_IN_4BIT = True

SYSTEM_PROMPT = (
    "You are an expert medical AI assistant. "
    "You will be provided with medical abstracts as context. "
    "Your task is to carefully read the context and use it to answer the user's question. "
    "Formulate a detailed explanation based on the context, and conclude with a "
    "final decision of Yes, No, or Maybe."
)


# ==========================================
# MODEL LOADING
# ==========================================
def load_model(model_path: str, use_merged: bool = False):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from: {model_path}")

    if use_merged:
        # Load standalone merged model (no base model needed)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=False,
        )
    else:
        # Load LoRA adapters on top of quantized base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
        )

    FastLanguageModel.for_inference(model)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Model loaded successfully.")
    return model, tokenizer


# ==========================================
# RAG RETRIEVAL
# ==========================================
def retrieve_context(question: str, top_k: int = 3):
    """Use the HybridRetriever to fetch relevant medical context."""
    try:
        from src.proposed.retriever import build_retriever

        retriever = build_retriever()
        docs = retriever.retrieve(question, final_top_k=top_k, use_rerank=True)
        context_parts = [d["text"] for d in docs]
        context_str = "\n\n".join(context_parts)
        print(f"[RAG] Retrieved {len(docs)} documents.")
        return context_str
    except Exception as e:
        print(f"[RAG] Warning: retrieval failed ({e}). Running without context.")
        return ""


# ==========================================
# INFERENCE
# ==========================================
def build_prompt(question: str, context: str, tokenizer) -> str:
    """Build the formatted prompt for the model."""
    if context:
        user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
    else:
        user_content = question

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def generate_answer(model, tokenizer, prompt: str) -> str:
    """Run model.generate() and return the decoded answer."""
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Strip the prompt tokens — decode only generated part
    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


# ==========================================
# INTERACTIVE LOOP
# ==========================================
def interactive_loop(model, tokenizer, use_rag: bool):
    """Simple REPL for continuous conversation."""
    print("\n" + "=" * 60)
    print("  Vietnamese Medical Chatbot — Interactive Mode")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        context = retrieve_context(question) if use_rag else ""
        prompt = build_prompt(question, context, tokenizer)

        print("Bot: ", end="", flush=True)
        answer = generate_answer(model, tokenizer, prompt)
        print(answer)
        print()


# ==========================================
# ENTRY POINT
# ==========================================
def main():
    global MAX_NEW_TOKENS
    parser = argparse.ArgumentParser(
        description="Vietnamese Medical Chatbot — CLI inference"
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Single question to answer. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to LoRA adapter or merged model dir. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--use_merged",
        action="store_true",
        help="Load as a merged 16-bit model instead of LoRA adapters.",
    )
    parser.add_argument(
        "--no_rag",
        action="store_true",
        help="Disable RAG retrieval (answer question directly without context).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Max tokens to generate. Default: {MAX_NEW_TOKENS}",
    )
    args = parser.parse_args()

    # Override global
    MAX_NEW_TOKENS = args.max_new_tokens

    use_rag = not args.no_rag

    # Load model
    model, tokenizer = load_model(args.model_path, use_merged=args.use_merged)

    if args.question:
        # Single-shot mode
        context = retrieve_context(args.question) if use_rag else ""
        prompt = build_prompt(args.question, context, tokenizer)
        answer = generate_answer(model, tokenizer, prompt)
        print("\n" + "=" * 60)
        print(f"Question: {args.question}")
        print("=" * 60)
        print(f"Answer:\n{answer}")
        print("=" * 60)
    else:
        # Interactive REPL
        interactive_loop(model, tokenizer, use_rag=use_rag)


if __name__ == "__main__":
    main()
