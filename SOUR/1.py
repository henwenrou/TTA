import re

input_file = "input.txt"
output_file = "cleaned.txt"

patterns = [
    r"^\[.*?\]\s*config\s*:\s*$",
    r"API Error: terminated",
    r"Interrupted",
    r"^Todos$",
    r"^[☒☐]",
    r"^⎿",
]

compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []

for line in lines:
    stripped = line.strip()

    # 超长异常行直接删
    if len(stripped) > 300:
        continue

    # 匹配垃圾日志
    if any(p.search(stripped) for p in compiled_patterns):
        continue

    cleaned_lines.append(line)

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(cleaned_lines)

print("清洗完成")