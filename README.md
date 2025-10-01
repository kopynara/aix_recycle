# aix_recycle 🗑️♻️

이미지 분류/전처리 및 모델 학습 프로젝트.  
팀 내에서 macOS, Linux, Windows 환경을 모두 지원합니다.

---

## 1. 환경 세팅

### macOS & Linux 🍎🐧
```bash
# 1) 가상환경 생성
python3 -m venv v_aix_recycle

# 2) 가상환경 활성화
source v_aix_recycle/bin/activate

# 3) pip 최신화 + 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

### Windows (PowerShell) 🪟
# 1) 가상환경 생성
python -m venv v_aix_recycle

# 2) 가상환경 활성화
.\v_aix_recycle\Scripts\activate

# 3) pip 최신화 + 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

## 📂 Dataset Guide
- 원본 데이터는 한글 폴더명 → 영어로 rename 후 사용
- 전체 매핑표는 [FOLDER_MAPPING.md](./FOLDER_MAPPING.md) 참고
