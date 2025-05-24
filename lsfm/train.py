import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

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
    parser.add_argument('--data_root',    type=str,   required=True,
                        help='Structured dataset root directory')
    parser.add_argument('--domain_list',  nargs='+',  required=True,
                        help='Domain names, e.g. F RF P RP')
    parser.add_argument('--train_txt',    type=str,   required=True,
                        help='Train split file (pid label list)')
    parser.add_argument('--val_txt',      type=str,   required=True,
                        help='Validation split file')
    parser.add_argument('--epochs',       type=int,   default=50,
                        help='Number of epochs for both style and adapt stages')
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--style_dim',    type=int,   default=64)
    parser.add_argument('--z_dim',        type=int,   default=16)
    parser.add_argument('--mmd_layers',   type=str,   default='2,4,6',
                        help='Comma-separated MMD layer indices')
    parser.add_argument('--lambda_mmd',   type=float, default=0.1)
    parser.add_argument('--save_dir',     type=str,   default='./checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--img_size',     type=int,   default=128,
                        help='Input image size')
    args = parser.parse_args()

    # epochs for both stages
    args.num_epochs_style = args.epochs
    args.num_epochs_adapt = args.epochs

    # parse mmd_layers
    args.mmd_layers = [int(x) for x in args.mmd_layers.split(',') if x]

    # infer num_domains and num_classes
    args.num_domains = len(args.domain_list)
    labels = set()
    with open(args.train_txt) as f:
        for l in f:
            parts = l.strip().split()
            if len(parts) >= 2:
                labels.add(int(parts[1]))
    args.num_classes = len(labels)

    os.makedirs(args.save_dir, exist_ok=True)
    return args

def default_transforms(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

# train.py 의 make_dataloaders() 부분
def make_dataloaders(args):
    tf = default_transforms(args.img_size)

    # StyleDataset: (data_root, domain_list, train_txt, transform)
    style_set = StyleDataset(
        args.data_root,
        args.domain_list,
        args.train_txt,
        transform=tf
    )
    style_loader = DataLoader(style_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)

    # AdaptDataset: unchanged
    adapt_set = AdaptDataset(
        args.data_root,
        args.domain_list[:-1],
        [args.domain_list[-1]],
        transform=tf
    )
    adapt_loader = DataLoader(adapt_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)

    return {'style': style_loader, 'adapt': adapt_loader}


def train_style(models, utils, dl, device, args):
    G, M, SE, D = models['G'], models['M'], models['SE'], models['D']
    F_ext, clf  = models['F_ext'], models['clf']
    optG, optM, optSE, optD, optF, optC = (
        utils['optG'], utils['optM'], utils['optSE'],
        utils['optD'], utils['optF'], utils['optC']
    )

    gan_loss = nn.MSELoss()
    cyc_loss = nn.L1Loss()
    sty_loss = nn.L1Loss()
    ce_loss  = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs_style):
        for real_s, real_t, src_id, ds, dt in dl['style']:
            real_s, real_t = real_s.to(device), real_t.to(device)
            src_id, ds, dt = src_id.to(device), ds.to(device), dt.to(device)

            # 1) Style transfer
            z    = torch.randn(real_s.size(0), args.z_dim, device=device)
            sc_t = M(z, dt)
            fake = G(real_s, sc_t)

            # 2) Discriminator 업데이트
            D_real = D(real_t, dt)
            D_fake = D(fake.detach(), dt)
            valid  = torch.ones_like(D_real)
            fake_l = torch.zeros_like(D_fake)
            lossD  = 0.5*(gan_loss(D_real, valid) + gan_loss(D_fake, fake_l))
            optD.zero_grad(); lossD.backward(); optD.step()

            # 3) Generator/Mapping/Encoder 업데이트
            Df2   = D(fake, dt)
            lossGg = gan_loss(Df2, valid)
            scp    = SE(fake, dt)
            lossS  = sty_loss(scp, sc_t)
            sc_s   = M(torch.randn_like(z), ds)
            rec    = G(fake, sc_s)
            lossC  = cyc_loss(rec, real_s)
            lossG  = lossGg + lossS + lossC
            optG.zero_grad(); optM.zero_grad(); optSE.zero_grad()
            lossG.backward()
            optG.step(); optM.step(); optSE.step()

            # 4) Classifier 파인튜닝
            feats, logits = clf(F_ext(fake.detach()))
            lossCls = ce_loss(logits, src_id)
            optF.zero_grad(); optC.zero_grad()
            lossCls.backward()
            optF.step(); optC.step()

        # 체크포인트 저장
        torch.save(G.state_dict(),      f"{args.save_dir}/G_s{epoch}.pth")
        torch.save(M.state_dict(),      f"{args.save_dir}/M_s{epoch}.pth")
        torch.save(SE.state_dict(),     f"{args.save_dir}/SE_s{epoch}.pth")
        torch.save(D.state_dict(),      f"{args.save_dir}/D_s{epoch}.pth")
        torch.save(F_ext.state_dict(),  f"{args.save_dir}/FE_s{epoch}.pth")
        torch.save(clf.state_dict(),    f"{args.save_dir}/C_s{epoch}.pth")
        print(f"[Style] Epoch {epoch} | D:{lossD:.4f} G:{lossG:.4f} C:{lossCls:.4f}")

def train_adapt(models, utils, dl, device, args):
    G, M       = models['G'], models['M']
    F_ext, clf = models['F_ext'], models['clf']
    optF, optC = utils['optF'], utils['optC']
    mmd_losser = MKMMDLoss(layer_ids=args.mmd_layers)

    for epoch in range(args.num_epochs_adapt):
        for real_s, s_id, ds, real_t, dt in dl['adapt']:
            real_s, real_t = real_s.to(device), real_t.to(device)
            s_id, ds, dt   = s_id.to(device), ds.to(device), dt.to(device)

            fake = G(real_s, M(torch.randn(real_s.size(0), args.z_dim, device=device), dt))
            fF, _ = clf(F_ext(fake))
            fR, _ = clf(F_ext(real_t))
            losses = [mmd_losser(x, y) for x, y in zip(fF, fR)]
            lossM  = sum(losses) * args.lambda_mmd

            optF.zero_grad(); optC.zero_grad()
            lossM.backward()
            optF.step(); optC.step()

        torch.save(F_ext.state_dict(), f"{args.save_dir}/FE_a{epoch}.pth")
        torch.save(clf.state_dict(),   f"{args.save_dir}/C_a{epoch}.pth")
        print(f"[Adapt] Epoch {epoch} | MMD:{lossM:.4f}")

def main():
    args    = parse_args()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = make_dataloaders(args)

    models = {
        'G':    Generator(style_dim=args.style_dim).to(device),
        'M':    MappingNetwork(latent_dim=args.z_dim,
                               style_dim=args.style_dim,
                               num_domains=args.num_domains).to(device),
        'SE':   StyleEncoder(img_size=args.img_size,
                             style_dim=args.style_dim,
                             num_domains=args.num_domains).to(device),
        'D':    Discriminator(in_channels=1,
                              num_domains=args.num_domains).to(device),
        'F_ext':FeatureExtractor(in_channels=1).to(device),
        'clf':  Classifier(in_dim=FeatureExtractor(in_channels=1).out_dim,
                           hidden_dims=[512],
                           num_classes=args.num_classes).to(device)
    }

    utils = {
        'optG':   optim.Adam(models['G'].parameters(),       lr=args.lr),
        'optM':   optim.Adam(models['M'].parameters(),       lr=args.lr),
        'optSE':  optim.Adam(models['SE'].parameters(),      lr=args.lr),
        'optD':   optim.Adam(models['D'].parameters(),       lr=args.lr),
        'optF':   optim.Adam(models['F_ext'].parameters(),   lr=args.lr),
        'optC':   optim.Adam(models['clf'].parameters(),     lr=args.lr),
    }

    train_style(models, utils, loaders, device, args)
    train_adapt(models, utils, loaders, device, args)

if __name__ == '__main__':
    main()
