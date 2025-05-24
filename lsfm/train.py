# ─────────────────────────────────────────────────────────
#  lsfm/train.py   (전체 파일 교체본)
# ─────────────────────────────────────────────────────────
import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

# ── 모델 / 데이터셋 임포트 ──────────────────────────────
from lsfm.models.generator         import Generator
from lsfm.models.mapping_net       import MappingNetwork
from lsfm.models.style_encoder     import StyleEncoder
from lsfm.models.discriminator     import Discriminator
from lsfm.models.feature_extractor import FeatureExtractor
from lsfm.models.classifier        import Classifier
from lsfm.models.mmd               import MKMMDLoss
from lsfm.datasets                 import StyleDataset, AdaptDataset
# ─────────────────────────────────────────────────────────


# --------------------------------------------------------
# 인자 파서
# --------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("LSFM Training")
    p.add_argument('--data_root',   type=str, required=True)
    p.add_argument('--domain_list', nargs='+', required=True)
    p.add_argument('--train_txt',   type=str, required=True)
    p.add_argument('--val_txt',     type=str, required=True)

    # 하이퍼파라미터
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--style_dim',  type=int, default=64)
    p.add_argument('--z_dim',      type=int, default=16)
    p.add_argument('--mmd_layers', type=str, default='2,4,6')
    p.add_argument('--lambda_mmd', type=float, default=0.1)
    p.add_argument('--img_size',   type=int, default=128)

    # 자동 추론 가능 항목
    p.add_argument('--num_domains', type=int, default=None)
    p.add_argument('--num_classes', type=int, default=None)

    # 기타
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    args = p.parse_args()

    args.mmd_layers = [int(x) for x in args.mmd_layers.split(',') if x]
    if args.num_domains is None:
        args.num_domains = len(args.domain_list)

    if args.num_classes is None:
        labels = set()
        with open(args.train_txt) as f:
            for line in f:
                _, lbl = line.strip().split()
                labels.add(int(lbl))
        args.num_classes = len(labels)

    return args


# --------------------------------------------------------
# 변환 파이프
# --------------------------------------------------------
def tfms(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])


# --------------------------------------------------------
# DataLoader 생성
# --------------------------------------------------------
def make_dataloaders(args):
    train_set = StyleDataset(
        root        = args.data_root,
        domain_list = args.domain_list,
        split_txt   = args.train_txt,        # ← 클래스 정의에 맞춰 이름 확인
        transform   = tfms(args.img_size)
    )
    val_set = StyleDataset(
        root        = args.data_root,
        domain_list = args.domain_list,
        split_txt   = args.val_txt,
        transform   = tfms(args.img_size)
    )
    style_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, drop_last=True
    )

    adapt_loader = DataLoader(
        AdaptDataset(
            root           = args.data_root,
            source_domains = args.domain_list[:-1],
            target_domains = [args.domain_list[-1]],
            transform      = tfms(args.img_size)
        ),
        batch_size=args.batch_size,
        shuffle=True, drop_last=True
    )
    return {'style': style_loader, 'adapt': adapt_loader}


# --------------------------------------------------------
# Style 단계 학습
# --------------------------------------------------------
def train_style(models, opts, loaders, device, args):
    G, M, SE, D = models['G'], models['M'], models['SE'], models['D']
    F_ext, clf  = models['F_ext'], models['clf']
    optG, optM, optSE = opts['G'], opts['M'], opts['SE']
    optD, optF, optC  = opts['D'], opts['F_ext'], opts['clf']

    l2 = nn.MSELoss()
    l1 = nn.L1Loss()
    ce = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        for real_s, real_t, lbl_s, ds, dt in loaders['style']:
            real_s, real_t = real_s.to(device), real_t.to(device)
            lbl_s, ds, dt  = lbl_s.to(device), ds.to(device), dt.to(device)

            z    = torch.randn(real_s.size(0), args.z_dim, device=device)
            s_t  = M(z, dt)
            fake = G(real_s, s_t)

            # --- D ---
            optD.zero_grad()
            valid = torch.ones_like(D(real_t, dt))
            fake0 = torch.zeros_like(valid)
            lossD = 0.5 * (l2(D(real_t, dt), valid) +
                           l2(D(fake.detach(), dt), fake0))
            lossD.backward(); optD.step()

            # --- G / M / SE ---
            optG.zero_grad(); optM.zero_grad(); optSE.zero_grad()
            lossG_adv = l2(D(fake, dt), valid)
            loss_sty  = l1(SE(fake, dt), s_t)
            rec       = G(fake, M(torch.randn_like(z), ds))
            loss_cyc  = l1(rec, real_s)
            (lossG_adv + loss_sty + loss_cyc).backward()
            optG.step(); optM.step(); optSE.step()

            # --- 분류기 파인튜닝 ---
            optF.zero_grad(); optC.zero_grad()
            feat = F_ext(fake.detach())
            _, logits = clf(feat)
            ce(logits, lbl_s).backward()
            optF.step(); optC.step()

        print(f"[Style] {ep:03} | D:{lossD.item():.4f}  G:{lossG_adv.item():.4f}")

        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(G.state_dict(),  f"{args.save_dir}/G_e{ep}.pth")
        torch.save(clf.state_dict(), f"{args.save_dir}/clf_e{ep}.pth")


# --------------------------------------------------------
# Adapt 단계 학습
# --------------------------------------------------------
def train_adapt(models, opts, loaders, device, args):
    G, M = models['G'], models['M']
    F_ext, clf = models['F_ext'], models['clf']
    optF, optC = opts['F_ext'], opts['clf']
    mmd = MKMMDLoss(layer_ids=args.mmd_layers)

    for ep in range(args.epochs):
        for s_img, _, ds, t_img, dt in loaders['adapt']:
            s_img, t_img = s_img.to(device), t_img.to(device)
            ds, dt       = ds.to(device), dt.to(device)

            fake = G(s_img, M(torch.randn(s_img.size(0), args.z_dim, device=device), dt))

            f_fake, _ = clf(F_ext(fake))
            f_real, _ = clf(F_ext(t_img))
            loss = sum(mmd(a, b) for a, b in zip(f_fake, f_real)) * args.lambda_mmd

            optF.zero_grad(); optC.zero_grad()
            loss.backward(); optF.step(); optC.step()

        print(f"[Adapt] {ep:03} | MMD:{loss.item():.4f}")


# --------------------------------------------------------
# main
# --------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- Data -----
    loaders = make_dataloaders(args)

    # ----- Model ----
    feat_net = FeatureExtractor(in_channels=1)
    models = {
        'G'     : Generator(style_dim=args.style_dim).to(device),
        'M'     : MappingNetwork(args.z_dim, args.style_dim, args.num_domains).to(device),
        'SE'    : StyleEncoder(args.img_size, args.style_dim, args.num_domains, in_channels=1).to(device),
        'D'     : Discriminator(in_channels=1, num_domains=args.num_domains).to(device),
        'F_ext' : feat_net.to(device),
        'clf'   : Classifier(feat_net.out_dim, [512], args.num_classes).to(device)
    }

    # ----- Optim ----
    opts = {
        'G'     : optim.Adam(models['G'].parameters(),  lr=args.lr),
        'M'     : optim.Adam(models['M'].parameters(),  lr=args.lr),
        'SE'    : optim.Adam(models['SE'].parameters(), lr=args.lr),
        'D'     : optim.Adam(models['D'].parameters(),  lr=args.lr),
        'F_ext' : optim.Adam(models['F_ext'].parameters(), lr=args.lr),
        'clf'   : optim.Adam(models['clf'].parameters(), lr=args.lr)
    }

    # ----- Run ------
    train_style(models, opts, loaders, device, args)
    train_adapt(models, opts, loaders, device, args)


if __name__ == '__main__':
    main()
# ─────────────────────────────────────────────────────────
