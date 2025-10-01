import os
import json
import csv
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ==============================
# 데이터 경로 설정
# ==============================
DATASET_PATH = Path("../datasets/processed/train")
OUTPUT_JSON = "class_weights.json"
OUTPUT_MD = "class_weights.md"

# ==============================
# 클래스별 샘플 수 세기
# ==============================
classes = sorted([d.name for d in DATASET_PATH.iterdir() if d.is_dir()])
sample_counts = {}

for cls in classes:
    cls_path = DATASET_PATH / cls
    sample_counts[cls] = len(list(cls_path.glob("*")))

print("클래스별 샘플 수:", sample_counts)

# ==============================
# class_weight 계산 (sklearn)
# ==============================
y = []
for cls, count in sample_counts.items():
    y.extend([cls] * count)  # 각 샘플을 클래스명으로 치환
y = np.array(y)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(classes),
    y=y
)

# dict 변환
class_weights_dict = {cls: float(w) for cls, w in zip(classes, class_weights)}

# ==============================
# 저장 (JSON)
# ==============================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(class_weights_dict, f, indent=4, ensure_ascii=False)

# ==============================
# 저장 (Markdown)
# ==============================
with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("# ⚖️ Class Weights Summary\n\n")
    f.write("| Class | Samples | Weight |\n")
    f.write("|-------|---------|--------|\n")
    for cls in classes:
        f.write(f"| {cls} | {sample_counts[cls]} | {class_weights_dict[cls]:.6f} |\n")

print(f"✅ 클래스 가중치 저장 완료 → {OUTPUT_JSON}, {OUTPUT_MD}")
