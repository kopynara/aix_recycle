import os
import shutil
import random
import logging
from pathlib import Path

import yaml
import splitfolders
from tqdm import tqdm

# -----------------------------
# Logging 설정
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                      # 콘솔 출력
        logging.FileHandler("prepare_data.log", "w")  # 로그 파일 저장
    ]
)

# -----------------------------
# Config 불러오기
# -----------------------------
def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# -----------------------------
# 데이터 분리 (비율 기반)
# -----------------------------
def prepare_and_split_data(config):
    input_dir = Path(config["dataset"]["raw_path"])
    output_dir = Path(config["dataset"]["processed_path"])
    ratio = tuple(config["dataset"]["split_ratio"])

    logging.info(f"입력 데이터 경로: {input_dir}")
    logging.info(f"출력 데이터 경로: {output_dir}")
    logging.info(f"분리 비율: {ratio}")

    # 1. 입력 데이터 확인
    if not input_dir.exists() or not any(input_dir.iterdir()):
        logging.error(f"❌ '{input_dir}' 폴더가 없거나 비어 있습니다.")
        input_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"👉 '{input_dir}' 폴더를 생성했습니다. 데이터를 넣고 다시 실행하세요.")
        return None

    # 2. 출력 데이터 확인
    if (output_dir / "train").exists() and (output_dir / "val").exists():
        logging.info(f"✅ '{output_dir}' 데이터셋이 이미 존재합니다. 기존 데이터를 사용합니다.")
        return str(output_dir)

    # 3. 새로 분리
    logging.info(f"✨ '{output_dir}' 폴더를 새로 생성하여 데이터셋을 분리합니다.")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        splitfolders.ratio(
            input_dir,
            output=str(output_dir),
            seed=42,
            ratio=ratio,
            move=False
        )
        logging.info("✨ 데이터 분리 완료!")

        # tqdm로 파일 개수 카운트
        for split in ["train", "val", "test"]:
            split_path = output_dir / split
            if split_path.exists():
                for class_dir in tqdm(list(split_path.iterdir()), desc=f"{split} 진행상황"):
                    num_files = len(list(class_dir.glob("*")))
                    logging.info(f"{split}/{class_dir.name}: {num_files}개 파일")

        return str(output_dir)

    except Exception as e:
        logging.error(f"❌ 데이터 분리 중 오류 발생: {e}")
        return None

# -----------------------------
# 데이터 소량 샘플링 (개수 기반)
# -----------------------------
def split_folder_fixed_count(config):
    ORIGINAL_DATA_PATH = Path(config["dataset"]["raw_path"])
    OUTPUT_DATA_PATH = Path("datasets/sample_dataset")

    NUM_TRAIN, NUM_VAL, NUM_TEST = 100, 10, 10  # 클래스별 최대 개수
    logging.info("샘플링 데이터셋을 생성합니다...")

    for split in ["train", "val", "test"]:
        (OUTPUT_DATA_PATH / split).mkdir(parents=True, exist_ok=True)

    for class_name in tqdm(list(ORIGINAL_DATA_PATH.iterdir()), desc="샘플링 진행중"):
        if not class_name.is_dir():
            continue

        all_files = [f for f in class_name.iterdir() if f.is_file()]
        random.shuffle(all_files)

        train_files = all_files[:NUM_TRAIN]
        val_files = all_files[len(train_files): len(train_files) + NUM_VAL]
        test_files = all_files[len(train_files) + len(val_files): len(train_files) + len(val_files) + NUM_TEST]

        for split_name, files in zip(["train", "val", "test"],
                                     [train_files, val_files, test_files]):
            dest_class_path = OUTPUT_DATA_PATH / split_name / class_name.name
            dest_class_path.mkdir(parents=True, exist_ok=True)
            for file in files:
                shutil.copy2(file, dest_class_path)

    logging.info(f"✅ 데이터셋 샘플링 완료 → '{OUTPUT_DATA_PATH}'")

# -----------------------------
# 실행 부분
# -----------------------------
if __name__ == "__main__":
    print("🚨 경고: 원본 파일은 삭제되지 않습니다. 안전하게 백업하세요. 🚨")

    config = load_config()

# 1) 비율로 나누기 (풀데이터)
prepare_and_split_data(config)

# 2) 소량 샘플링 (빠른 학습 테스트용 → 필요시 주석 해제)
# split_folder_fixed_count(config)
