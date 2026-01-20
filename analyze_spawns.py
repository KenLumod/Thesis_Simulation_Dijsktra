
import re

try:
    with open('debug_output.txt', 'r', encoding='utf-16') as f:
        content = f.read()
except:
    try:
        with open('debug_output.txt', 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        print("Could not read file.")
        exit()

# Find all mappings
mappings = re.findall(r"Spawn (\d+) mapped to Cell (\d+)", content)
warnings = re.findall(r"Warning: No road cell found", content)

print(f"Total Mapped: {len(mappings)}")
print(f"Total Warnings: {len(warnings)}")

# Check duplicates
cell_counts = {}
for sid, cid in mappings:
    cell_counts[cid] = cell_counts.get(cid, 0) + 1

print(f"Unique Cells Mapped: {len(cell_counts)}")
duplicates = {k: v for k, v in cell_counts.items() if v > 1}
if duplicates:
    print(f"Duplicate Mappings (Road Cell ID -> Count): {duplicates}")
else:
    print("No duplicate mappings from different spawns.")
