import sys
import os
sys.path.append("/home/lad/AI/vietnamese-medical-chatbot")
import re
from benchmark import extract_mcq_answer

log_path = "/home/lad/.gemini/antigravity/brain/a05eaca0-42af-4967-9e24-3678b9ca9809/.system_generated/tasks/task-237.log"
with open(log_path, "r", encoding="utf-8") as f:
    log_content = f.read()

# Let's split by "=================================================="
parts = log_content.split("==================================================")

generation_idx = 0
for part in parts:
    part_clean = part.strip()
    if not part_clean:
        continue
    if "Question:" not in part_clean and "Context:" not in part_clean and "Evaluating MedQA" not in part_clean:
        generation_idx += 1
        pred = extract_mcq_answer(part_clean)
        print(f"Gen {generation_idx}: pred = {pred}")
        print(f"Snippet: {repr(part_clean[:120])}... [len: {len(part_clean)}]")
