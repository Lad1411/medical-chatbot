import sys
import os
sys.path.append("/home/lad/AI/vietnamese-medical-chatbot")
import re
from datasets import load_dataset

def improved_extract_mcq_answer(llm_output):
    # Strategy 0: Look at the very beginning of the output
    # Matches "A\n\n...", "B\n\n...", "C) ...", etc.
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

log_path = "/home/lad/.gemini/antigravity/brain/a05eaca0-42af-4967-9e24-3678b9ca9809/.system_generated/tasks/task-237.log"
with open(log_path, "r", encoding="utf-8") as f:
    log_content = f.read()

parts = log_content.split("==================================================")
generations = []
for part in parts:
    part_clean = part.strip()
    if not part_clean:
        continue
    if "Question:" not in part_clean and "Context:" not in part_clean and "Evaluating MedQA" not in part_clean and "FINAL RESULTS" not in part_clean and "Writing results" not in part_clean and "Unsloth" not in part_clean and "MEDICAL BENCHMARK" not in part_clean:
        generations.append(part_clean)

# Load MedQA dataset expected answers
medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
dataset = medqa_ds.select(range(10))
actual_answers = [item["answer_idx"] for item in dataset]

print(f"Found {len(generations)} generations. Expecting 10.")
correct = 0
for i, gen in enumerate(generations[:10]):
    pred = improved_extract_mcq_answer(gen)
    expected = actual_answers[i]
    is_correct = (pred == expected)
    if is_correct:
        correct += 1
    print(f"Q{i+1}: expected={expected}, pred={pred}, Correct={is_correct}")
    # Print the first line of the generation to verify
    first_line = gen.split("\n")[0]
    print(f"   First line: {repr(first_line)}")

print(f"Accuracy with improved extraction: {correct/10 * 100:.2f}%")
