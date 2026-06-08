import sys

log_path = "/home/lad/.gemini/antigravity/brain/a05eaca0-42af-4967-9e24-3678b9ca9809/.system_generated/tasks/task-237.log"
with open(log_path, "r", encoding="utf-8") as f:
    log_content = f.read()

# Let's find "[*] Evaluating MedQA (10 questions)..."
eval_start_marker = "[*] Evaluating MedQA (10 questions)..."
start_idx = log_content.find(eval_start_marker)
if start_idx == -1:
    print("Not found start.")
    sys.exit(1)

content = log_content[start_idx:]
# Let's split by the 50 equals separator:
separator = "=================================================="
blocks = content.split(separator)

print(f"Total blocks after split: {len(blocks)}")
for i, block in enumerate(blocks):
    block_clean = block.strip()
    if not block_clean:
        continue
    # Let's print the first 2 lines and the block index
    lines = block_clean.split("\n")
    print(f"\n--- Block {i} (Lines: {len(lines)}) ---")
    print("First line:", repr(lines[0]))
    if len(lines) > 1:
        print("Second line:", repr(lines[1]))
    if len(lines) > 5:
        print("Last line:", repr(lines[-1]))
