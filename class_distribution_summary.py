import os
import csv
from pathlib import Path
from collections import defaultdict

# 데이터셋 경로
DATASET_PATH = Path("../datasets/processed")
OUTPUT_CSV = "class_distribution_summary.csv"
OUTPUT_MD = "class_distribution_summary.md"

splits = ["train", "val", "test"]
summary = defaultdict(dict)

# 각 split별 클래스 개수 카운트
for split in splits:
    split_path = DATASET_PATH / split
    if not split_path.exists():
        continue
    for class_dir in sorted(split_path.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*")))
            summary[class_dir.name][split] = count

# CSV 저장
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["Class", "Train", "Val", "Test"]
    writer.writerow(header)
    for cls, counts in summary.items():
        writer.writerow([cls, counts.get("train", 0), counts.get("val", 0), counts.get("test", 0)])

# Markdown 저장
with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("# 📊 클래스 분포 요약\n\n")
    f.write("> 🚨 test는 무시하세요 (split_ratio=8:2:0 설정인데 splitfolders 특성상 최소 0~1개 생성됨)\n\n")
    f.write("| Class | Train | Val | Test |\n")
    f.write("|-------|-------|-----|------|\n")
    for cls, counts in summary.items():
        f.write(f"| {cls} | {counts.get('train', 0)} | {counts.get('val', 0)} | {counts.get('test', 0)} |\n")

print(f"✅ Summary 저장 완료 → {OUTPUT_CSV}, {OUTPUT_MD}")
