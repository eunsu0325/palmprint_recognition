import os
import sys
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

# --- 모델 임포트 ---
from lsfm.models.generator         import Generator
from lsfm.models.mapping_net       import MappingNetwork
from lsfm.models.style_encoder     import StyleEncoder
from lsfm.models.discriminator     import Discriminator
from lsfm.models.feature_extractor import FeatureExtractor
from lsfm.models.classifier        import Classifier
from lsfm.models.mmd               import MKMMDLoss
from lsfm.datasets                 import StyleDataset, AdaptDataset


def parse_args():
    parser = argparse.ArgumentParser(description="LSFM Training Script")
    parser.add_argument('--img_size', type=int, default=128,
                        help='Input image size (e.g. 128 or 256)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--style_dim', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--num_domains', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=250)
    parser.add_argument('--num_epochs_style', type=int, default=50)
    parser.add_argument('--num_epochs_adapt', type=int, default=100)
    parser.add_argument('--lambda_mmd', type=float, default=0.1)
    args, _ = parser.parse_known_args()
    return args


def default_transforms(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])


def train_style(models, optimizers, dataloaders, device, args):
    G, M, SE, D = models['G'], models['M'], models['SE'], models['D']
    F_ext, clf = models['F_ext'], models['clf']
    opt_G, opt_M, opt_SE, opt_D, opt_F, opt_clf = (
        optimizers['opt_G'], optimizers['opt_M'], optimizers['opt_SE'], 
        optimizers['opt_D'], optimizers['opt_F'], optimizers['opt_clf']
    )

    # 손실 정의: Least-Squares GAN (LSGAN)
    criterion_gan = nn.MSELoss()
    criterion_cyc = nn.L1Loss()
    criterion_sty = nn.L1Loss()
    criterion_ce  = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs_style):
        for real_s, real_t, src_id, ds, dt in dataloaders['style']:
            real_s, real_t = real_s.to(device), real_t.to(device)
            src_id, ds, dt = src_id.to(device), ds.to(device), dt.to(device)

            # 1) 스타일 전송
            z     = torch.randn(real_s.size(0), args.z_dim, device=device)
            sc_t  = M(z, dt)
            fake_t = G(real_s, sc_t)

            # 2) D 업데이트 (LSGAN)
            D_real = D(real_t, dt)
            D_fake = D(fake_t.detach(), dt)
            valid  = torch.ones_like(D_real)
            fake   = torch.zeros_like(D_fake)
            loss_D = 0.5 * (criterion_gan(D_real, valid) + criterion_gan(D_fake, fake))
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # 3) G, M, SE 업데이트
            D_fake2    = D(fake_t, dt)
            loss_G_gan = criterion_gan(D_fake2, valid)
            sc_pred    = SE(fake_t, dt)
            loss_sty   = criterion_sty(sc_pred, sc_t)
            z2    = torch.randn_like(z)
            sc_s  = M(z2, ds)
            rec_s = G(fake_t, sc_s)
            loss_cyc = criterion_cyc(rec_s, real_s)
            loss_G = loss_G_gan + loss_sty + loss_cyc
            opt_G.zero_grad(); opt_M.zero_grad(); opt_SE.zero_grad()
            loss_G.backward()
            opt_G.step(); opt_M.step(); opt_SE.step()

            # 4) 분류기 파인튜닝 (Eq.8)
            feat_fake = F_ext(fake_t.detach())
            _, logits_fake = clf(feat_fake)
            loss_cls = criterion_ce(logits_fake, src_id)
            opt_F.zero_grad(); opt_clf.zero_grad()
            loss_cls.backward()
            opt_F.step(); opt_clf.step()

        # 체크포인트 저장
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(G.state_dict(),  os.path.join(args.output_dir, f'G_epoch{epoch}.pth'))
        torch.save(M.state_dict(),  os.path.join(args.output_dir, f'M_epoch{epoch}.pth'))
        torch.save(SE.state_dict(), os.path.join(args.output_dir, f'SE_epoch{epoch}.pth'))
        torch.save(D.state_dict(),  os.path.join(args.output_dir, f'D_epoch{epoch}.pth'))
        torch.save(F_ext.state_dict(), os.path.join(args.output_dir, f'F_ext_epoch{epoch}.pth'))
        torch.save(clf.state_dict(),   os.path.join(args.output_dir, f'clf_epoch{epoch}.pth'))
        print(f"[Style] Epoch {epoch:02d} | loss_D: {loss_D:.4f} | loss_G: {loss_G:.4f} | loss_cls: {loss_cls:.4f}")


def train_adapt(models, optimizers, dataloaders, device, args):
    F_ext, clf = models['F_ext'], models['clf']
    G, M       = models['G'], models['M']
    opt_F, opt_clf = optimizers['opt_F'], optimizers['opt_clf']
    criterion_mmd = MKMMDLoss()

    for epoch in range(args.num_epochs_adapt):
        for real_s, s_id, ds, real_t, dt in dataloaders['adapt']:
            real_s, real_t = real_s.to(device), real_t.to(device)
            s_id, ds, dt   = s_id.to(device), ds.to(device), dt.to(device)

            # 1) fake target 생성
            z      = torch.randn(real_s.size(0), args.z_dim, device=device)
            sc_t   = M(z, dt)
            fake_t = G(real_s, sc_t)

            # 2) 특징 추출
            feats_fake, _ = clf(F_ext(fake_t))
            feats_real, _ = clf(F_ext(real_t))

            # 3) MMD 손실
            loss_mmd = sum(criterion_mmd(f1, f2) for f1, f2 in zip(feats_fake, feats_real))
            loss_mmd *= args.lambda_mmd

            # 4) FE + clf 업데이트
            opt_F.zero_grad(); opt_clf.zero_grad()
            loss_mmd.backward()
            opt_F.step(); opt_clf.step()

        torch.save(F_ext.state_dict(), os.path.join(args.output_dir, f'F_ext_adapt_epoch{epoch}.pth'))
        torch.save(clf.state_dict(),   os.path.join(args.output_dir, f'clf_adapt_epoch{epoch}.pth'))
        print(f"[Adapt] Epoch {epoch:02d} | MMD: {loss_mmd:.4f}")


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domain_list    = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    source_domains = domain_list[:-1]
    target_domains = domain_list[-1:]

    transforms = default_transforms(args.img_size)
    dataloaders = {
        'style': DataLoader(
            StyleDataset(args.data_root, domain_list, transform=transforms),
            batch_size=args.batch_size, shuffle=True, drop_last=True
        ),
        'adapt': DataLoader(
            AdaptDataset(args.data_root, source_domains, target_domains, transform=transforms),
            batch_size=args.batch_size, shuffle=True, drop_last=True
        )
    }

    models = {
        'G': Generator(img_size=args.img_size, style_dim=args.style_dim).to(device),
        'M': MappingNetwork(latent_dim=args.z_dim, style_dim=args.style_dim, num_domains=args.num_domains).to(device),
        'SE': StyleEncoder(img_size=args.img_size, in_channels=1, style_dim=args.style_dim, num_domains=args.num_domains).to(device),
        'D': Discriminator(in_channels=1, num_domains=args.num_domains).to(device),
        'F_ext': FeatureExtractor(in_channels=1).to(device),
        'clf': Classifier(in_dim=FeatureExtractor(in_channels=1).out_dim, hidden_dims=[512], num_classes=args.num_classes).to(device)
    }

    optimizers = {
        'opt_G':   optim.Adam(models['G'].parameters(),     lr=args.lr),
        'opt_M':   optim.Adam(models['M'].parameters(),     lr=args.lr),
        'opt_SE':  optim.Adam(models['SE'].parameters(),    lr=args.lr),
        'opt_D':   optim.Adam(models['D'].parameters(),     lr=args.lr),
        'opt_F':   optim.Adam(models['F_ext'].parameters(), lr=args.lr),
        'opt_clf': optim.Adam(models['clf'].parameters(),  lr=args.lr),
    }

    train_style(models, optimizers, dataloaders, device, args)
    train_adapt(models, optimizers, dataloaders, device, args)


if __name__ == '__main__':
    main()
