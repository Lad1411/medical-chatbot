import sys
import os
sys.path.append("/home/lad/AI/vietnamese-medical-chatbot")
import re
from datasets import load_dataset
from benchmark import extract_mcq_answer

log_path = "/home/lad/.gemini/antigravity/brain/a05eaca0-42af-4967-9e24-3678b9ca9809/.system_generated/tasks/task-237.log"
with open(log_path, "r", encoding="utf-8") as f:
    log_content = f.read()

# Let's find "Evaluating MedQA" in the log
start_idx = log_content.find("[*] Evaluating MedQA")
if start_idx == -1:
    print("Could not find start of evaluation in log.")
    sys.exit(1)

eval_content = log_content[start_idx:]

# Split by the separator
parts = eval_content.split("==================================================")

# Let's filter parts.
generations = []
for p in parts:
    p_clean = p.strip()
    if not p_clean:
        continue
    # Let's check if it has the prompt structure
    if "Question:" in p_clean:
        # This is a prompt
        continue
    if "Evaluating MedQA" in p_clean or "MedQA:" in p_clean or "FINAL RESULTS" in p_clean or "Writing results" in p_clean or "Done." in p_clean:
        continue
    generations.append(p_clean)

# Load MedQA expected answers
medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
dataset = medqa_ds.select(range(10))
actual_answers = [item["answer_idx"] for item in dataset]

print(f"Filtered actual generations: {len(generations)}")
correct = 0

def improved_extract_mcq_answer(llm_output):
    # Strategy 0: Look at the very beginning of the output
    match_start = re.match(r"^\s*([A-D])(?:\b|\)|\]|\.)", llm_output, re.IGNORECASE)
    if match_start:
        return match_start.group(1).upper()
        
    match_mcq = re.search(r"(?:Answer:|answer is|correct option is|choice is|final answer:)\s*([A-D])", llm_output, re.IGNORECASE)
    if match_mcq:
        return match_mcq.group(1).upper()
        
    match_mcq_end = re.search(r"\b([A-D])\b[\.\s]*(?:<\|im_end\|>)?$", llm_output, re.IGNORECASE)
    if match_mcq_end:
        return match_mcq_end.group(1).upper()
        
    return None

for i in range(min(len(generations), 10)):
    gen = generations[i]
    expected = actual_answers[i]
    pred_original = extract_mcq_answer(gen)
    pred_improved = improved_extract_mcq_answer(gen)
    is_correct_improved = (pred_improved == expected)
    if is_correct_improved:
        correct += 1
    first_line = gen.split('\n')[0]
    print(f"Q{i+1}: expected={expected}, original_pred={pred_original}, improved_pred={pred_improved}, Correct={is_correct_improved}")
    print(f"   First line of output: {repr(first_line)}")

print(f"Improved Accuracy: {correct/10 * 100:.2f}%")
