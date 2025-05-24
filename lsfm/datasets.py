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
    스타일 매칭(stage1) 용 데이터셋.
    train_txt 에 정의된 훈련 샘플만 로드해서,
    랜덤 source/target 쌍을 반환.
    반환값: real_s, real_t, src_id, ds_idx, dt_idx
    """
    def __init__(
        self,
        data_root:   str,
        domain_list: list[str],
        train_txt:   str,
        transform=None
    ):
        super().__init__()
        self.data_root = data_root
        self.domains   = domain_list
        self.transform = transform or default_transforms()

        # train_txt 파일을 읽어서, domain 별 (경로, 라벨) 리스트로 저장
        self.images = {d: [] for d in self.domains}
        with open(train_txt, 'r') as f:
            for line in f:
                relpath, lbl = line.strip().split()
                dom = relpath.split('/')[0]
                if dom not in self.domains:
                    continue
                full = os.path.join(self.data_root, relpath)
                if os.path.isfile(full):
                    self.images[dom].append((full, int(lbl)))

        # 전체 샘플 수
        self.length = sum(len(v) for v in self.images.values())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 랜덤 source/target 도메인 선택 (서로 달라야 함)
        ds = random.choice(self.domains)
        dt_choices = [d for d in self.domains if d != ds]
        dt = random.choice(dt_choices)

        # 각 도메인에서 랜덤 샘플링
        src_path, src_id = random.choice(self.images[ds])
        tgt_path, _      = random.choice(self.images[dt])

        real_s = Image.open(src_path).convert('L')
        real_t = Image.open(tgt_path).convert('L')
        if self.transform:
            real_s = self.transform(real_s)
            real_t = self.transform(real_t)

        return real_s, real_t, src_id, self.domains.index(ds), self.domains.index(dt)


class AdaptDataset(Dataset):
    """
    특성 정합(stage2) 용 데이터셋.
    하나의 source 이미지를 identity, ds_idx 와 함께,
    하나의 target 이미지와 dt_idx 를 반환.
    """
    def __init__(
        self,
        data_root:      str,
        source_domains: list[str],
        target_domains: list[str],
        transform=None
    ):
        super().__init__()
        self.data_root   = data_root
        self.src_domains = source_domains
        self.tgt_domains = target_domains
        self.transform   = transform or default_transforms()

        # source 이미지 목록
        self.src_images = []
        for d in self.src_domains:
            dir_d = os.path.join(self.data_root, d)
            for pid in sorted(os.listdir(dir_d)):
                pdir = os.path.join(dir_d, pid)
                if not os.path.isdir(pdir): continue
                for fn in os.listdir(pdir):
                    full = os.path.join(pdir, fn)
                    self.src_images.append((full, int(pid), self.src_domains.index(d)))

        # target 이미지 목록
        self.tgt_images = []
        for d in self.tgt_domains:
            dir_d = os.path.join(self.data_root, d)
            for pid in sorted(os.listdir(dir_d)):
                pdir = os.path.join(dir_d, pid)
                if not os.path.isdir(pdir): continue
                for fn in os.listdir(pdir):
                    full = os.path.join(pdir, fn)
                    self.tgt_images.append((full, self.tgt_domains.index(d)))

        # 순회 가능하도록 둘 중 더 큰 길이 사용
        self.length = max(len(self.src_images), len(self.tgt_images))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s_path, s_id, ds = self.src_images[idx % len(self.src_images)]
        t_path, dt       = self.tgt_images[idx % len(self.tgt_images)]

        real_s = Image.open(s_path).convert('L')
        real_t = Image.open(t_path).convert('L')
        if self.transform:
            real_s = self.transform(real_s)
            real_t = self.transform(real_t)

        return real_s, s_id, ds, real_t, dt
