# LSFM: Light Style and Feature Matching for Efficient Cross-Domain Palmprint Recognition

> **논문**: LSFM – Light-StarGAN 기반 스타일 매칭 및 Multi-Kernel MMD를 이용한 효율적 교차 도메인 손바닥 인식  
> **저자**: Ruan et al. (2024)  
> **코드 구현**: 개인 구현 기반

---

## 📌 프로젝트 개요

LSFM은 LightStarGAN 기반의 경량 스타일 전송 네트워크와 Multi-Kernel MMD 기반 특징 정합을 결합하여, 도메인 간 손바닥 인식 정확도를 높이는 비지도 도메인 적응(UDA) 모델입니다.

- 스타일 매칭: ShuffleNetV2 기반 Generator + AdaIN
- 특징 정합: VGG16 기반 Feature Extractor + Classifier + MK-MMD
- 도메인 분포 차이를 극복하며, 새로운 환경에서도 손바닥 인식 가능

---

## 📁 폴더 구조

```
lsfm/
├── datasets.py
├── train.py
├── eval.py
├── requirements.txt
└── models/
    ├── generator.py
    ├── mapping_net.py
    ├── style_encoder.py
    ├── discriminator.py
    ├── feature_extractor.py
    ├── classifier.py
    ├── mmd.py
    └── model_blocks.py
```

---

## ⚙️ 환경 설정

```bash
pip install -r requirements.txt
```

`requirements.txt`에는 다음과 같은 패키지가 포함되어야 합니다:

```
torch
torchvision
numpy
pillow
scikit-learn
munch
tqdm
```

---

## 🗂 데이터 준비

```
data_root/
├── DomainA/
│   ├── 0/
│   ├── 1/
│   └── ...
├── DomainB/
└── DomainC/  # 타깃 도메인
```

각 도메인 폴더는 identity별 하위 폴더로 구성되어 있어야 합니다.

---

## 🚀 학습 실행

```bash
python train.py \
  --img_size 128 \
  --data_root /path/to/data_root \
  --output_dir ./checkpoints \
  --batch_size 32 \
  --lr 1e-4 \
  --style_dim 64 \
  --z_dim 16 \
  --num_domains 3 \
  --num_classes 250 \
  --num_epochs_style 50 \
  --num_epochs_adapt 100 \
  --lambda_mmd 0.1
```

---

## 🧪 평가 실행

```bash
python eval.py \
  --img_size 128 \
  --data_root /path/to/data_root \
  --checkpoint_dir ./checkpoints \
  --batch_size 32 \
  --num_classes 250 \
  --target_domains DomainC
```

---

## 🧩 모듈 설명

- `generator.py`: ShuffleNet 기반 Generator
- `mapping_net.py`: 도메인 조건화 Mapping Network
- `style_encoder.py`: StarGAN v2 기반 Style Encoder
- `discriminator.py`: 도메인 조건 multi-head Discriminator
- `feature_extractor.py`: VGG16 기반 특성 추출기
- `classifier.py`: multi-FC classifier (layer-wise output)
- `mmd.py`: MK-MMD (Multi-kernel Maximum Mean Discrepancy) 손실
- `datasets.py`: 도메인-아이디 기반 스타일/적응 데이터셋

---

## 📝 License

This code is provided for **non-commercial research purposes** under the **CC BY-NC 4.0 license**.
