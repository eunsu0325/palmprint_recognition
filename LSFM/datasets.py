# lsfm/datasets.py
import os, random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def default_transforms(img_size=128):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

class StyleDataset(Dataset):
    """
    Dataset for style matching stage. On each sample, randomly pick a source
    and a target domain (different), then sample one image from each.
    Returns:
      real_s, real_t, label_s (identity), ds (source domain idx), dt (target domain idx)
    """
    def __init__(self, data_root, domain_list, transform=None):
        super().__init__()
        self.data_root = data_root
        self.domains = domain_list
        self.transform = transform or default_transforms()
        # Build list of (img_path, identity) per domain
        self.images = {}
        for d in self.domains:
            img_dir = os.path.join(data_root, d)
            # assume subfolders per identity
            paths = []
            for id_name in sorted(os.listdir(img_dir)):
                id_dir = os.path.join(img_dir, id_name)
                if not os.path.isdir(id_dir): continue
                for fname in os.listdir(id_dir):
                    paths.append((os.path.join(id_dir, fname), int(id_name)))
            self.images[d] = paths
        # flatten for length
        self.length = sum(len(v) for v in self.images.values())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Randomly pick source and target domains
        ds = random.choice(self.domains)
        dt = random.choice([d for d in self.domains if d != ds])
        # Randomly sample one image from each
        src_path, src_id = random.choice(self.images[ds])
        tgt_path, _ = random.choice(self.images[dt])
        real_s = Image.open(src_path).convert('L')
        real_t = Image.open(tgt_path).convert('L')
        if self.transform:
            real_s = self.transform(real_s)
            real_t = self.transform(real_t)
        return real_s, real_t, src_id, self.domains.index(ds), self.domains.index(dt)

class AdaptDataset(Dataset):
    """
    Dataset for adaptation (feature matching). Returns one source image,
    its identity label and domain idx, and one target image & domain idx.
    """
    def __init__(self, data_root, source_domains, target_domains, transform=None):
        super().__init__()
        self.data_root = data_root
        self.src_domains = source_domains
        self.tgt_domains = target_domains
        self.transform = transform or default_transforms()
        # Build lists similarly
        self.src_images = []
        for d in self.src_domains:
            img_dir = os.path.join(data_root, d)
            for id_name in sorted(os.listdir(img_dir)):
                id_dir = os.path.join(img_dir, id_name)
                if not os.path.isdir(id_dir): continue
                for fname in os.listdir(id_dir):
                    self.src_images.append((os.path.join(id_dir, fname), int(id_name), self.src_domains.index(d)))
        self.tgt_images = []
        for d in self.tgt_domains:
            img_dir = os.path.join(data_root, d)
            for id_name in sorted(os.listdir(img_dir)):
                id_dir = os.path.join(img_dir, id_name)
                if not os.path.isdir(id_dir): continue
                for fname in os.listdir(id_dir):
                    self.tgt_images.append((os.path.join(id_dir, fname), self.tgt_domains.index(d)))
        # length is max of two to allow cycling
        self.length = max(len(self.src_images), len(self.tgt_images))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s_path, s_id, ds = self.src_images[idx % len(self.src_images)]
        t_path, dt = self.tgt_images[idx % len(self.tgt_images)]
        real_s = Image.open(s_path).convert('L')
        real_t = Image.open(t_path).convert('L')
        if self.transform:
            real_s = self.transform(real_s)
            real_t = self.transform(real_t)
        return real_s, s_id, ds, real_t, dt
