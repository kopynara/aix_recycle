# flatten_224.py
import os
import shutil
from pathlib import Path

base_dir = Path("~/projects/datasets/raw/train").expanduser()

for class_dir in base_dir.iterdir():
    if class_dir.is_dir():
        sub_224 = class_dir / "224"
        if sub_224.exists() and sub_224.is_dir():
            print(f"📂 Flatten: {sub_224} → {class_dir}")
            for file in sub_224.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(class_dir / file.name))
            # 빈 224 폴더 삭제
            sub_224.rmdir()
