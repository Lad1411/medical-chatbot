"""
benchmark.py — RAG Benchmark Suite (MedQA Only)
=================================================
Runs only RAG-based evaluation phases in sequence:

  Phase 4 │ MedQA │ CoT-200 LoRA │ BM25 retriever
  Phase 5 │ MedQA │ CoT-200 LoRA │ Dense retriever
  Phase 6 │ MedQA │ CoT-200 LoRA │ Hybrid + Rerank

Usage:
  python benchmark.py \
      --cot200-lora /path/to/cot200_lora \
      [--limit 100] \
      [--output results.csv] \
      [--phases all]          # or e.g. "4,5" to run specific phases
"""

import argparse
import csv
import re
import gc
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import logging as hf_logging
from unsloth import FastLanguageModel
from retriever import build_retriever, FINAL_TOP_K

# ==========================================
# PERFORMANCE SETTINGS
# ==========================================
torch.backends.cuda.matmul.allow_tf32 = True
hf_logging.set_verbosity_error()

# ==========================================
# CONFIGURATION
# ==========================================
BASE_MODEL          = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_CONTEXT_TOKENS  = 1536   # matches training max_seq_length in trainer.py
MAX_NEW_TOKENS      = 768    # enough for CoT reasoning + conclusion
BATCH_SIZE          = 16

# ==========================================
# SYSTEM PROMPTS
# ==========================================
SYS_MEDQA_NO_RAG = (
    "You are an expert medical AI assistant. Your task is to answer the multiple choice question. "
    "Formulate a detailed explanation using your medical knowledge, then conclude with: "
    "Conclusion: A, B, C, or D."
)

SYS_MEDQA_RAG = (
    "You are an expert medical AI assistant. You will be provided with medical reference context. "
    "Carefully read the context and use it to answer the multiple choice question. "
    "Formulate a detailed explanation based on the context, then conclude with: "
    "Conclusion: A, B, C, or D."
)

SYS_PUBMED_BUILTIN = (
    "You are a helpful and expert medical assistant. "
    "You will be provided with a medical research abstract as context. "
    "Carefully read the context and use it to evaluate the question. "
    "Conclude your reasoning with a final classification of 'yes', 'no', or 'maybe'."
)

# ==========================================
# PHASE DEFINITIONS  (RAG phases only)
# ==========================================
PHASES = [
    {
        "id":             4,
        "name":           "MedQA | CoT-200 + BM25",
        "model_key":      "cot200",
        "dataset":        "medqa",
        "retriever_mode": "bm25",
    },
    {
        "id":             5,
        "name":           "MedQA | CoT-200 + Dense",
        "model_key":      "cot200",
        "dataset":        "medqa",
        "retriever_mode": "dense",
    },
    {
        "id":             6,
        "name":           "MedQA | CoT-200 + Hybrid + Rerank",
        "model_key":      "cot200",
        "dataset":        "medqa",
        "retriever_mode": "hybrid",
    },
]

# ==========================================
# ARGUMENT PARSER
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="RAG Medical Benchmark Suite (MedQA, Phases 4-6)")
    parser.add_argument(
        "--cot200-lora", type=str, default="",
        help="Path to CoT-200 LoRA adapter (required for Phases 4-6)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of questions per dataset (0 = no limit, use full set)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.csv",
        help="Output CSV file for detailed per-question results"
    )
    parser.add_argument(
        "--phases", type=str, default="all",
        help="Comma-separated phase IDs to run, e.g. '4,5' or 'all'"
    )
    return parser.parse_args()

# ==========================================
# ANSWER EXTRACTION
# ==========================================
def extract_mcq_answer(text: str):
    """Extract A/B/C/D from generated text."""
    # Primary: look for explicit conclusion/answer prefix
    match = re.search(
        r"(?:Conclusion:|Answer:|answer is|correct (?:answer|option|choice) is)\s*([A-D])",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()
    # Fallback: last standalone letter at end of text
    match_end = re.search(
        r"\b([A-D])\b[\.\s]*(?:<\|im_end\|>)?$",
        text, re.IGNORECASE
    )
    if match_end:
        return match_end.group(1).upper()
    return None

def extract_ynm_answer(text: str):
    """Extract yes/no/maybe from generated text."""
    # Primary: look for explicit conclusion/answer prefix
    match = re.search(
        r"(?:Conclusion:|Answer:|answer is|classification is)\s*(yes|no|maybe)",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).lower()
    # Fallback: last standalone yes/no/maybe at end of text
    match_end = re.search(
        r"\b(yes|no|maybe)\b[\.\s]*(?:<\|im_end\|>)?$",
        text, re.IGNORECASE
    )
    if match_end:
        return match_end.group(1).lower()
    return None

# ==========================================
# RETRIEVAL DISPATCH
# ==========================================
def format_context(hits: list) -> str:
    if not hits:
        return "No reference context found."
    docs = []
    for i, hit in enumerate(hits):
        title = hit.get("metadata", {}).get("title", f"Document {i+1}")
        text  = hit.get("text", "")
        docs.append(f"[Document {i+1} — {title}]\n{text}")
    return "\n\n".join(docs)

def retrieve_context(query: str, retriever, mode: str) -> str:
    """
    Dispatch retrieval to the correct method based on mode.
    Returns empty string for 'none' and 'builtin' modes.
    """
    if mode in ("none", "builtin") or retriever is None:
        return ""
    if mode == "bm25":
        hits = retriever._bm25_search(query, top_k=FINAL_TOP_K)
    elif mode == "dense":
        hits = retriever._dense_search(query, top_k=FINAL_TOP_K)
    elif mode == "hybrid":
        hits = retriever.retrieve(query, final_top_k=FINAL_TOP_K, use_rerank=True)
    else:
        hits = []
    return format_context(hits)

# ==========================================
# PROMPT BUILDING
# ==========================================
def _truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

def build_medqa_prompt(question: str, options: dict, context: str,
                       tokenizer, retriever_mode: str) -> str:
    options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
    user_question = f"Question:\n{question}\n\nOptions:\n{options_text}"

    use_rag   = retriever_mode != "none"
    sys_prompt = SYS_MEDQA_RAG if use_rag else SYS_MEDQA_NO_RAG

    if use_rag and context:
        sys_len = len(tokenizer.encode(sys_prompt))
        q_len   = len(tokenizer.encode(user_question))
        budget  = MAX_CONTEXT_TOKENS - sys_len - q_len - 50
        context = _truncate_to_tokens(context, tokenizer, max(budget, 0))
        user_content = f"Context:\n{context}\n\n{user_question}"
    else:
        user_content = user_question

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def build_pubmedqa_prompt(question: str, abstract: str, tokenizer) -> str:
    sys_len = len(tokenizer.encode(SYS_PUBMED_BUILTIN))
    q_len   = len(tokenizer.encode(question))
    budget  = MAX_CONTEXT_TOKENS - sys_len - q_len - 50
    abstract = _truncate_to_tokens(abstract, tokenizer, max(budget, 0))

    user_content = f"Context:\n{abstract}\n\nQuestion: {question}\nAnswer:"
    messages = [
        {"role": "system", "content": SYS_PUBMED_BUILTIN},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# ==========================================
# BATCH GENERATION
# ==========================================
def generate_batch(model, tokenizer, prompts: list) -> list:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS,
    )
    inputs      = {k: v.to(model.device) for k, v in inputs.items()}
    input_len   = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.05,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[:, input_len:]
    decoded    = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [t.strip() for t in decoded]

# ==========================================
# MODEL MANAGEMENT
# ==========================================
def load_model(lora_path: str = None):
    label = f"LoRA: {lora_path}" if lora_path else "base (no LoRA)"
    print(f"\n[+] Loading model — {label}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_CONTEXT_TOKENS,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path:
        model.load_adapter(lora_path)

    FastLanguageModel.for_inference(model)
    model.eval()
    print("[+] Model ready.")
    return model, tokenizer

def unload_model(model, tokenizer):
    print("[+] Unloading model from GPU...")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# PHASE RUNNERS
# ==========================================
def run_pubmedqa_phase(phase: dict, model, tokenizer, dataset,
                       results_list: list) -> tuple:
    """
    PubMedQA — uses the abstract already embedded in each dataset row.
    No external retriever needed.
    """
    correct, total = 0, len(dataset)
    print(f"\n[*] Phase {phase['id']}: {phase['name']}  ({total} questions)")

    for start in tqdm(range(0, total, BATCH_SIZE), desc="PubMedQA", colour="green"):
        try:
            batch    = dataset[start : start + BATCH_SIZE]
            prompts, actuals, questions = [], [], []

            for question, ctx_obj, actual in zip(
                batch["question"], batch["context"], batch["final_decision"]
            ):
                # Ground-truth abstract is already inside the dataset row
                abstract = "\n\n".join(ctx_obj["contexts"])
                prompt   = build_pubmedqa_prompt(question, abstract, tokenizer)
                prompts.append(prompt)
                actuals.append(actual.strip().lower())
                questions.append(question)

            outputs = generate_batch(model, tokenizer, prompts)

            for i, text in enumerate(outputs):
                pred       = extract_ynm_answer(text)
                is_correct = (pred == actuals[i])
                if is_correct:
                    correct += 1
                results_list.append({
                    "Phase":     phase["name"],
                    "Dataset":   "PubMedQA",
                    "Model":     phase["model_key"],
                    "Retriever": phase["retriever_mode"],
                    "Index":     start + i + 1,
                    "Question":  questions[i][:200],
                    "Expected":  actuals[i],
                    "Predicted": pred if pred else "[Invalid]",
                    "IsCorrect": is_correct,
                })

        except Exception as e:
            print(f"  [!] Batch {start} skipped: {e}")
            continue

    acc = correct / total * 100 if total > 0 else 0.0
    print(f"  → Accuracy: {acc:.2f}%  ({correct}/{total})")
    return correct, total


def run_medqa_phase(phase: dict, model, tokenizer, dataset,
                    retriever, results_list: list) -> tuple:
    """
    MedQA — supports four retriever modes: none / bm25 / dense / hybrid.
    """
    correct, total = 0, len(dataset)
    mode           = phase["retriever_mode"]
    print(f"\n[*] Phase {phase['id']}: {phase['name']}  ({total} questions)")

    for start in tqdm(range(0, total, BATCH_SIZE), desc=f"MedQA/{mode}", colour="cyan"):
        try:
            batch    = dataset[start : start + BATCH_SIZE]
            prompts, actuals, questions, contexts = [], [], [], []

            for question, options, actual in zip(
                batch["question"], batch["options"], batch["answer_idx"]
            ):
                options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
                query        = f"{question}\n{options_text}"
                context      = retrieve_context(query, retriever, mode)

                prompt = build_medqa_prompt(question, options, context, tokenizer, mode)
                prompts.append(prompt)
                actuals.append(actual)
                questions.append(question)
                contexts.append(context)

            outputs = generate_batch(model, tokenizer, prompts)

            for i, text in enumerate(outputs):
                pred       = extract_mcq_answer(text)
                is_correct = (str(pred).upper() == str(actuals[i]).upper())
                if is_correct:
                    correct += 1
                results_list.append({
                    "Phase":     phase["name"],
                    "Dataset":   "MedQA",
                    "Model":     phase["model_key"],
                    "Retriever": mode,
                    "Index":     start + i + 1,
                    "Question":  questions[i][:200],
                    "Expected":  actuals[i],
                    "Predicted": pred if pred else "[Invalid]",
                    "IsCorrect": is_correct,
                })

        except Exception as e:
            print(f"  [!] Batch {start} skipped: {e}")
            continue

    acc = correct / total * 100 if total > 0 else 0.0
    print(f"  → Accuracy: {acc:.2f}%  ({correct}/{total})")
    return correct, total

# ==========================================
# MAIN
# ==========================================
def main():
    args = parse_args()

    # ── Resolve which phases to run ──────────────────────────────────
    if args.phases.strip().lower() == "all":
        phases_to_run = PHASES
    else:
        selected = {int(x.strip()) for x in args.phases.split(",")}
        phases_to_run = [p for p in PHASES if p["id"] in selected]

    if not phases_to_run:
        print("[!] No valid phases selected. Exiting.")
        return

    # ── Validate LoRA paths ──────────────────────────────────────────
    lora_map = {
        "cot200": args.cot200_lora if args.cot200_lora else None,
    }
    for phase in phases_to_run:
        key = phase["model_key"]
        if not lora_map.get(key):
            print(f"[!] Phase {phase['id']} requires --{key}-lora but it was not provided.")
            return

    print("=" * 65)
    print("  MEDICAL BENCHMARK SUITE  (RAG phases only)")
    print("=" * 65)
    print(f"  Phases to run : {[p['id'] for p in phases_to_run]}")
    print(f"  CoT-200 LoRA  : {lora_map['cot200']}")
    print(f"  Limit         : {args.limit if args.limit > 0 else 'full dataset'}")
    print(f"  Output        : {args.output}")
    print("=" * 65)

    # ── Build retriever (always needed for RAG phases) ───────────────
    print("\n[+] Initializing retriever (BM25 + Dense + Reranker)...")
    retriever = build_retriever()
    print("[+] Retriever ready.")

    # ── Load MedQA dataset ───────────────────────────────────────────
    print("\n[+] Loading MedQA dataset...")
    medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    if args.limit > 0:
        medqa_ds = medqa_ds.select(range(min(args.limit, len(medqa_ds))))
    print(f"  MedQA : {len(medqa_ds)} questions")

    # ── Run all phases (reload model only when model_key changes) ────
    all_results    = []
    phase_summaries = []

    current_model_key = None
    model, tokenizer  = None, None

    for phase in phases_to_run:
        # Reload model when the required model changes
        if phase["model_key"] != current_model_key:
            if model is not None:
                unload_model(model, tokenizer)
            model, tokenizer  = load_model(lora_map[phase["model_key"]])
            current_model_key = phase["model_key"]

        print(f"\n{'─' * 65}")
        print(f"  ▶ Phase {phase['id']}: {phase['name']}")
        print(f"{'─' * 65}")

        correct, total = run_medqa_phase(
            phase, model, tokenizer, medqa_ds, retriever, all_results
        )

        acc = correct / total * 100 if total > 0 else 0.0
        phase_summaries.append({
            "Phase":    phase["id"],
            "Name":     phase["name"],
            "Correct":  correct,
            "Total":    total,
            "Accuracy": f"{acc:.2f}%",
        })

    # Unload the last model
    if model is not None:
        unload_model(model, tokenizer)

    # ── Print final summary ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'#':<4}  {'Accuracy':<10}  {'Score':<14}  Name")
    print(f"  {'─'*4}  {'─'*10}  {'─'*14}  {'─'*30}")
    for s in phase_summaries:
        score = f"{s['Correct']}/{s['Total']}"
        print(f"  {s['Phase']:<4}  {s['Accuracy']:<10}  {score:<14}  {s['Name']}")
    print("=" * 65)

    # ── Save detailed CSV ────────────────────────────────────────────
    print(f"\n[+] Saving detailed results → {args.output}")
    if all_results:
        with open(args.output, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    # ── Save summary CSV ─────────────────────────────────────────────
    summary_path = args.output.replace(".csv", "_summary.csv")
    print(f"[+] Saving summary       → {summary_path}")
    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Phase", "Name", "Correct", "Total", "Accuracy"])
        writer.writeheader()
        writer.writerows(phase_summaries)

    print("\n[+] All done!")


if __name__ == "__main__":
    main()