import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, roc_curve
import numpy as np
import torchvision.transforms as T

# --- 모델 임포트 ---
from lsfm.models.feature_extractor import FeatureExtractor
from lsfm.models.classifier        import Classifier


def parse_args():
    parser = argparse.ArgumentParser(description="LSFM Evaluation Script")
    parser.add_argument('--img_size',       type=int,   default=128,
                        help='Input image size (must match training)')
    parser.add_argument('--data_root',      type=str,   required=True,
                        help='Dataset root directory')
    parser.add_argument('--checkpoint_dir', type=str,   required=True,
                        help='Directory with saved F_ext_best.pth and clf_best.pth')
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--num_classes',    type=int,   required=True,
                        help='Number of identity classes')
    parser.add_argument('--target_domains', nargs='+', required=True,
                        help='List of target domain folder names')
    args = parser.parse_args()
    return args


def default_transforms(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])


class TestDataset(Dataset):
    """
    Real target-domain dataset for evaluation.
    Returns: (image_tensor, identity_label)
    """
    def __init__(self, data_root, domains, transform=None):
        super().__init__()
        self.transform = transform
        self.items = []
        for d in domains:
            dom_dir = os.path.join(data_root, d)
            if not os.path.isdir(dom_dir):
                continue
            for id_name in sorted(os.listdir(dom_dir)):
                id_dir = os.path.join(dom_dir, id_name)
                if not os.path.isdir(id_dir):
                    continue
                label = int(id_name)
                for fname in os.listdir(id_dir):
                    self.items.append((os.path.join(id_dir, fname), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label


def compute_eer(y_true, y_score):
    # y_true: binary labels 0/1, y_score: score for positive class
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    F_ext = FeatureExtractor(in_channels=1).to(device)
    clf   = Classifier(in_dim=F_ext.out_dim,
                       hidden_dims=[512],
                       num_classes=args.num_classes).to(device)

    F_ext.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'F_ext_best.pth')))
    clf.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'clf_best.pth')))
    F_ext.eval(); clf.eval()

    # Prepare DataLoader
    transforms = default_transforms(args.img_size)
    test_ds = TestDataset(args.data_root, args.target_domains, transform=transforms)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            feats = F_ext(imgs)
            _, logits = clf(feats)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            scores = probs[torch.arange(len(labels)), labels]
            y_score.extend(scores.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    y_true_bin = [int(p == t) for p, t in zip(y_pred, y_true)]
    eer = compute_eer(y_true_bin, y_score)

    print(f"Identification Accuracy: {acc*100:.2f}%")
    print(f"Verification EER: {eer*100:.2f}%")


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
