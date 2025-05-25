# ───────────────────────────────────────── lsfm/eval.py (교정본)
import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, roc_curve
import numpy as np
import torchvision.transforms as T

# --- 모델 ---
from lsfm.models.feature_extractor import FeatureExtractor
from lsfm.models.classifier        import Classifier


# ───────────────────────────── argparse
def parse_args():
    p = argparse.ArgumentParser("LSFM Evaluation")
    p.add_argument('--data_root',      required=True)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--epoch',   type=int, required=True)
    p.add_argument('--target_domains', nargs='+', required=True)
    p.add_argument('--num_classes', type=int, required=True)
    p.add_argument('--img_size',   type=int, default=128)
    p.add_argument('--batch_size', type=int, default=32)
    return p.parse_args()


# ───────────────────────────── dataset util
def tfms(sz):
    return T.Compose([
        T.Resize((sz, sz)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

class TestSet(Dataset):
    def __init__(self, root, domains, transform):
        self.items, self.t = [], transform
        for d in domains:
            ddir = os.path.join(root, d)
            if not os.path.isdir(ddir): continue
            for pid in sorted(os.listdir(ddir)):
                id_dir = os.path.join(ddir, pid)
                if not os.path.isdir(id_dir) or not pid.isdigit(): continue
                label = int(pid)
                for f in os.listdir(id_dir):
                    self.items.append((os.path.join(id_dir, f), label))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, lab = self.items[i]
        img = Image.open(path).convert('L')
        return self.t(img), lab


# ───────────────────────────── metrics
def eer_score(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return 0.5 * (fpr[idx] + fnr[idx])


# ───────────────────────────── main eval
def main():
    a   = parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1️⃣ 모델 로드 ------------------------------------------------------------
    F = FeatureExtractor(1).to(dev)
    C = Classifier(in_dim=F.out_dim, hidden_dims=[512], num_classes=a.num_classes).to(dev)

    F.load_state_dict(torch.load(os.path.join(a.checkpoint_dir, f'FE_a{a.epoch}.pth')))
    C.load_state_dict(torch.load(os.path.join(a.checkpoint_dir, f'C_a{a.epoch}.pth')))
    F.eval(); C.eval()

    # 2️⃣ 데이터 ---------------------------------------------------------------
    ds = TestSet(a.data_root, a.target_domains, tfms(a.img_size))
    ld = DataLoader(ds, batch_size=a.batch_size, shuffle=False)

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, label in ld:
            x, label = x.to(dev), label.to(dev)

            # Skip 샘플: 라벨이 classifier 범위 밖 ➜ 기록만
            in_range_mask = label < a.num_classes
            if not in_range_mask.any():
                continue
            x, label = x[in_range_mask], label[in_range_mask]

            logits = C(F(x))[1]            # (B, num_classes)
            prob   = torch.softmax(logits, 1)
            pred   = prob.argmax(1)

            y_true.extend(label.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            # torch.gather 로 정해진 정답 클래스 확률 추출
            correct_prob = prob.gather(1, label.unsqueeze(1)).squeeze(1)
            y_prob.extend(correct_prob.cpu().tolist())

    # 3️⃣ 지표 -----------------------------------------------------------------
    acc = accuracy_score(y_true, y_pred) * 100.0
    # binary success/failure
    y_bin = [int(p == t) for p, t in zip(y_pred, y_true)]
    eer = eer_score(y_bin, y_prob) * 100.0

    print(f"Identification Accuracy : {acc:.2f}%")
    print(f"Verification EER        : {eer:.2f}%")

if __name__ == '__main__':
    main()
