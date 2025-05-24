# ───────────────────────────────────────────────────────────────
#  lsfm/train.py  (LSFM 논문 구조에 맞춘 전체 교체본)
# ───────────────────────────────────────────────────────────────
import os, argparse, torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from lsfm.models.generator     import Generator
from lsfm.models.mapping_net   import MappingNetwork
from lsfm.models.style_encoder import StyleEncoder
from lsfm.models.discriminator import Discriminator
from lsfm.models.feature_extractor import FeatureExtractor
from lsfm.models.classifier    import Classifier
from lsfm.models.mmd           import MKMMDLoss
from lsfm.datasets             import StyleDataset, AdaptDataset

# -----------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # data ---------------------------------------------------------
    p.add_argument('--data_root',   required=True)
    p.add_argument('--domain_list', nargs='+', required=True)
    p.add_argument('--train_txt',   required=True)
    p.add_argument('--val_txt',     required=True)
    # hyper --------------------------------------------------------
    p.add_argument('--epochs',      type=int, default=50)
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--style_dim',   type=int,   default=64)
    p.add_argument('--z_dim',       type=int,   default=16)
    p.add_argument('--img_size',    type=int,   default=128)
    p.add_argument('--mmd_layers',  default='2,4,6')
    p.add_argument('--lambda_mmd',  type=float, default=0.1)
    p.add_argument('--save_dir',    default='./checkpoints')
    # optional -----------------------------------------------------
    p.add_argument('--num_domains', type=int)
    p.add_argument('--num_classes', type=int)
    args = p.parse_args()

    args.mmd_layers = [int(i) for i in args.mmd_layers.split(',') if i]

    if args.num_domains is None:
        args.num_domains = len(args.domain_list)
    if args.num_classes is None:
        with open(args.train_txt) as f:
            args.num_classes = len({int(l.split()[1]) for l in f})
    return args


# -----------------------------------------------------------------
def tfms(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])


# -----------------------------------------------------------------
def make_dataloaders(args):
    train_set = StyleDataset(args.data_root,
                             args.domain_list,
                             args.train_txt,
                             transform=tfms(args.img_size))
    val_set   = StyleDataset(args.data_root,
                             args.domain_list,
                             args.val_txt,
                             transform=tfms(args.img_size))
    style_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    adapt_loader = DataLoader(
        AdaptDataset(args.data_root,
                     source_domains=args.domain_list[:-1],
                     target_domains=[args.domain_list[-1]],
                     transform=tfms(args.img_size)),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    return {'style': style_loader, 'adapt': adapt_loader}


# -----------------------------------------------------------------
def build_models(args, device):
    G   = Generator(style_dim=args.style_dim).to(device)
    M   = MappingNetwork(args.z_dim, args.style_dim,
                         args.num_domains).to(device)
    SE  = StyleEncoder(args.img_size, args.style_dim,
                       args.num_domains, in_channels=1).to(device)
    D   = Discriminator(in_channels=1,
                        num_domains=args.num_domains,
                        last_kernel=2).to(device)   # ← 4→2 kernel 변경!

    F   = FeatureExtractor(in_channels=1).to(device)
    clf = Classifier(in_dim=F.out_dim,
                     hidden_dims=[512],
                     num_classes=args.num_classes,
                     return_features=True).to(device)   # logits & feats
    return {'G': G, 'M': M, 'SE': SE, 'D': D, 'F_ext': F, 'clf': clf}


# -----------------------------------------------------------------
def build_optimizers(models, lr):
    return {k: optim.Adam(v.parameters(), lr=lr)
            for k, v in models.items() if k != 'SE'} | \
           {'opt_SE': optim.Adam(models['SE'].parameters(), lr=lr)}


# -----------------------------------------------------------------
def style_stage(models, opt, loader, device, args):
    G, M, SE, D, F_ext, clf = (models[k] for k in
                               ('G', 'M', 'SE', 'D', 'F_ext', 'clf'))
    mse, l1, ce = nn.MSELoss(), nn.L1Loss(), nn.CrossEntropyLoss()

    for real_s, real_t, src_id, ds, dt in loader:
        real_s, real_t = real_s.to(device), real_t.to(device)
        src_id, ds, dt = src_id.to(device), ds.to(device), dt.to(device)

        z = torch.randn(real_s.size(0), args.z_dim, device=device)
        sc_t  = M(z, dt)
        fake  = G(real_s, sc_t)

        # Discriminator ------------------------------------------
        D_real, D_fake = D(real_t, dt), D(fake.detach(), dt)
        loss_D = 0.5 * (mse(D_real, torch.ones_like(D_real)) +
                        mse(D_fake, torch.zeros_like(D_fake)))
        opt['opt_D'].zero_grad(); loss_D.backward(); opt['opt_D'].step()

        # Generator ---------------------------------------------
        loss_G_gan = mse(D(fake, dt), torch.ones_like(D_fake))
        loss_sty   = l1(SE(fake, dt), sc_t)

        rec = G(fake, M(torch.randn_like(z), ds))
        loss_cyc = l1(rec, real_s)

        loss_G = loss_G_gan + loss_sty + loss_cyc
        for k in ('opt_G', 'opt_M', 'opt_SE'):
            opt[k].zero_grad()
        loss_G.backward()
        for k in ('opt_G', 'opt_M', 'opt_SE'):
            opt[k].step()

        # Classifier fine-tune ----------------------------------
        feats = F_ext(fake.detach())
        _, logits = clf(feats)
        loss_cls = ce(logits, src_id)
        opt['opt_F_ext'].zero_grad(); opt['opt_clf'].zero_grad()
        loss_cls.backward()
        opt['opt_F_ext'].step(); opt['opt_clf'].step()


# -----------------------------------------------------------------
def adapt_stage(models, opt, loader, device, args):
    G, M, F_ext, clf = (models[k] for k in ('G', 'M', 'F_ext', 'clf'))
    mmd = MKMMDLoss(layer_ids=args.mmd_layers)

    for real_s, _, ds, real_t, dt in loader:
        real_s, real_t = real_s.to(device), real_t.to(device)
        ds, dt = ds.to(device), dt.to(device)

        fake = G(real_s, M(torch.randn(real_s.size(0),
                                       args.z_dim,
                                       device=device), dt))

        feats_f, _ = clf(F_ext(fake))
        feats_r, _ = clf(F_ext(real_t))
        loss = sum(mmd(a, b) for a, b in zip(feats_f, feats_r))
        loss *= args.lambda_mmd

        opt['opt_F_ext'].zero_grad(); opt['opt_clf'].zero_grad()
        loss.backward()
        opt['opt_F_ext'].step(); opt['opt_clf'].step()


# -----------------------------------------------------------------
def save_ckpt(models, save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)
    for k, m in models.items():
        torch.save(m.state_dict(), os.path.join(save_dir, f'{k}_{tag}.pth'))


# -----------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaders = make_dataloaders(args)
    models  = build_models(args, device)
    optims  = build_optimizers(models, args.lr)

    # -----------------------------------------------------------
    for ep in range(args.epochs):
        models['G'].train(); models['clf'].train()

        style_stage(models, optims, loaders['style'], device, args)
        adapt_stage(models, optims, loaders['adapt'], device, args)

        if (ep + 1) % 5 == 0:
            save_ckpt(models, args.save_dir, f'ep{ep+1}')
            print(f"[epoch {ep+1:03d}] checkpoint saved")

    print("✅  Training finished")


if __name__ == "__main__":
    main()
