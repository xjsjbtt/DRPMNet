#!/usr/bin/env python3
"""
滑坡分割数据加载器（按原 DiGATe-UNet 训练风格预处理）

对齐仓库里 `dataset/bijie_6c.py::TwoComposites(bands='RGB&DEM')` 的处理逻辑：
- resize（训练/验证/测试一致）
- 生成双流：RGB 与 DEM(复制到 3 通道后再做增强)
- 每通道 min-max 归一化到 [0,1]
- 训练集增强：`dataset.transforms.DualStreamTransform`（翻转 + 噪声 + 椒盐 + CLAHE）

说明：该风格不会做 ImageNet normalize / color jitter / 旋转缩放。
"""

import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

from dataset.transforms import DualStreamTransform


# =====================================================
# 全局随机种子（只在最外层调用一次）
# =====================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================

# =====================================================
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =====================================================
# Dataset
# =====================================================
class LandslideDataset(Dataset):

    def __init__(
        self,
        data_dir,
        split="train",
        image_size=224,
        augment=True,
        noise_std=0.1,
    ):
        self.split = split
        self.size = int(image_size)
        self.augment = augment and split == "train"
        self.noise_std = noise_std

        data_dir = Path(data_dir)
        self.image_dir = data_dir / split / "image"
        self.label_dir = data_dir / split / "label"
        self.dem_dir   = data_dir / split / "dem"

        self.image_files = sorted(self.image_dir.glob("*.png"))
        print(f"[{split}] found {len(self.image_files)} samples")

        # 原模型训练风格增强：仅训练集启用
        self.dual_transform = DualStreamTransform(p=0.5) if self.augment else None

    def _minmax_norm_per_channel(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        对齐 `TwoComposites.__normalize`：每个通道单独做 min-max 到 [0,1] 并 clamp。
        x: [C,H,W]
        """
        x = x.clone()
        for i in range(x.shape[0]):
            ch = x[i]
            mn = ch.min()
            mx = ch.max()
            if mx > mn:
                ch = (ch - mn) / (mx - mn + eps)
            x[i] = torch.clamp(ch, 0.0, 1.0)
        return x

    # -------------------------------------------------
    # __getitem__
    # -------------------------------------------------
    def __getitem__(self, idx):
        name = self.image_files[idx].name

        img = Image.open(self.image_dir / name).convert("RGB")
        lbl = Image.open(self.label_dir / name).convert("L")
        dem = Image.open(self.dem_dir   / name).convert("L")

        # 对齐 TwoComposites：仅 resize（几何增强不在这里做）
        img = F.resize(img, (self.size, self.size), InterpolationMode.BILINEAR)
        dem = F.resize(dem, (self.size, self.size), InterpolationMode.BILINEAR)
        lbl = F.resize(lbl, (self.size, self.size), InterpolationMode.NEAREST)

        img_tensor = F.to_tensor(img)
        dem_tensor = F.to_tensor(dem)

        # 对齐 TwoComposites：每通道 min-max 归一化
        img_tensor = self._minmax_norm_per_channel(img_tensor)
        dem_tensor = self._minmax_norm_per_channel(dem_tensor)

        lbl_np = np.array(lbl, dtype=np.int64)
        lbl_tensor = torch.from_numpy(lbl_np)
        lbl_tensor = (lbl_tensor > 0).long()

        # 对齐 DualStreamTransform：训练集在 tensor 级别做增强（同步作用在两路输入与标签）
        if self.dual_transform is not None:
            y = lbl_tensor.float()  # DualStreamTransform 里按 float 处理
            x2 = dem_tensor.repeat(3, 1, 1)  # 与 TwoComposites 一致：DEM 复制成 3 通道参与增强
            img_tensor, x2, y = self.dual_transform(img_tensor, x2, y)
            dem_tensor = x2[:1]  # 仍然输出 1 通道 dem，训练端会在需要时复制到 3 通道
            lbl_tensor = (y > 0.5).long()

        return {
            "image": img_tensor,
            "dem": dem_tensor,
            "label": lbl_tensor,
            "filename": name
        }

    def __len__(self):
        return len(self.image_files)


# =====================================================
# DataLoader 创建（科研规范）
# =====================================================
def create_data_loaders(
    data_dir,
    batch_size=8,
    num_workers=4,
    image_size=224,
    train_augment=True,
    val_augment=False,
    test_augment=False,
    noise_std=0.1,
    seed=42,
):
    seed_everything(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset = LandslideDataset(
        data_dir,
        split="train",
        image_size=image_size,
        augment=train_augment,
        noise_std=noise_std,
    )

    val_dataset = LandslideDataset(
        data_dir,
        split="val",
        image_size=image_size,
        augment=val_augment,
        noise_std=noise_std,
    )

    test_dataset = LandslideDataset(
        data_dir,
        split="test",
        image_size=image_size,
        augment=test_augment,
        noise_std=noise_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, test_loader, val_dataset, test_dataset
