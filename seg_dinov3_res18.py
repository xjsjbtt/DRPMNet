#!/usr/bin/env python3
"""
DINOv3 图像分割模型（引入 ResNet DEM 提取 + UPer 风格融合解码）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov3.hub.backbones import dinov3_vits16, dinov3_vitl16, Weights as BackboneWeights
from torchvision import models
from pathlib import Path


class ConvBNAct(nn.Sequential):
    """卷积 + BN + ReLU:基础构建，为了通道对齐、特征平滑、非线性增强"""

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

#上采样加1*1conv，用在decoder，attention前
class QUpsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.drop = nn.Dropout2d(p=0.1)
    def forward(self, q):
        q = F.interpolate(
            q, scale_factor=2, mode="bilinear", align_corners=False
        )
        q = self.proj(q)
        q = self.drop(q)
        return q

class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        q_channels,
        kv_channels,
        embed_dim,
        num_heads
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim#256
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(q_channels, embed_dim)
        self.k_proj = nn.Linear(kv_channels, embed_dim)
        self.v_proj = nn.Linear(kv_channels, embed_dim)

        self.out_proj = nn.Linear(embed_dim, q_channels)
        self.norm = nn.LayerNorm(q_channels)

    def forward(self, q, kv):
        """
        q :  [B, Cq, H, W]
        kv:  [B, Ck, H, W]
        """
        B, Cq, H, W = q.shape
        N = H * W

        q_in = q

        # -------- flatten --------
        q = q.flatten(2).transpose(1, 2)     # [B, N, Cq]
        kv = kv.flatten(2).transpose(1, 2)   # [B, N, Ck]

        # 线性变换生成qkv
        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        # Q，K，V分成多头
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数，用soft得到权重
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = attn.softmax(dim=-1)
        #注意力权重加权value
        out = attn_weights @ V  # [B, heads, N, head_dim]

        #合并多头结果
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, N, self.embed_dim)

        out = self.out_proj(out)  # [B, N, Cq]

        #
        out = out.transpose(1, 2).reshape(B, Cq, H, W)
        out = self.norm((out + q_in).flatten(2).transpose(1, 2))#跳跃连接
        out = out.transpose(1, 2).reshape(B, Cq, H, W)

        return out

class CrossAttnStage(nn.Module):
    def __init__(
        self,
        q_channels,
        kv_channels,
        embed_dim,
        num_heads,
        upsample
    ):
        super().__init__()

        self.upsample = QUpsample(q_channels) if upsample else None

        self.attn = MultiHeadCrossAttention(
            q_channels=q_channels,
            kv_channels=kv_channels,
            embed_dim=embed_dim,
            num_heads=num_heads
        )

    def forward(self, q, kv):
        if self.upsample is not None:
            q = self.upsample(q)
        q = self.attn(q, kv)
        return q



class DEMResNetExtractor(nn.Module):
    """
    ResNet18 DEM encoder
    输出:
        c1: [B, 64,  H/4,  W/4 ]
        c2: [B, 128, H/8,  W/8 ]
        c3: [B, 256, H/16, W/16]
    """
    def __init__(self, pretrained=True):
        super().__init__()
        net = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # 改成单通道输入
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 保存stem和前3个阶段
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # 输出64
        self.layer2 = net.layer2  # 输出128
        self.layer3 = net.layer3  # 输出256

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)  # H/4
        c2 = self.layer2(c1) # H/8
        c3 = self.layer3(c2) # H/16
        return c1, c2, c3



class PyramidCrossAttnDecoder(nn.Module):
    """三阶段 Cross-Attention: q16 → q8 → q4"""
    def __init__(self):
        super().__init__()
        self.stage1 = CrossAttnStage(
            q_channels=512,   # Q16通道
            kv_channels=256,  # c3
            embed_dim=256,
            num_heads=8,
            upsample=False
        )
        self.stage2 = CrossAttnStage(
            q_channels=512,
            kv_channels=128,  # c2
            embed_dim=256,
            num_heads=8,
            upsample=True
        )
        self.stage3 = CrossAttnStage(
            q_channels=512,
            kv_channels=64,   # c1
            embed_dim=256,
            num_heads=8,
            upsample=True
        )

    def forward(self, q16, c3, c2, c1):
        q16 = self.stage1(q16, c3)
        q8  = self.stage2(q16, c2)
        q4  = self.stage3(q8, c1)
        return q4



class DRFusionHead(nn.Module):
    def __init__(self, img_ch=1024, num_classes=1):
        super().__init__()
        # 将 ViT 输出 1024 → 512
        self.img_proj = ConvBNAct(img_ch, 512, k=1, p=0)
        self.decoder = PyramidCrossAttnDecoder()
        self.head = nn.Sequential(
            ConvBNAct(512, 128, k=3, p=1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, img_feat, dem_feats, img_size):
        c1, c2, c3 = dem_feats
        q16 = self.img_proj(img_feat)
        q4 = self.decoder(q16, c3, c2, c1)  # 三阶段
        logits = self.head(q4)
        logits = F.interpolate(logits, size=img_size, mode="bilinear", align_corners=False)
        return logits





class DINOv3SegmentationModel(nn.Module):
    def __init__(self, num_classes, pretrained_path=None, freeze_backbone=True):
        super().__init__()
        self.num_classes = num_classes

        # DINOv3 backbone
        if pretrained_path and Path(pretrained_path).exists():
            print(f"加载 DINOv3 预训练权重: {pretrained_path}")
            self.backbone = dinov3_vitl16(pretrained=False)  # <- 注意这里 False
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            missing_keys, unexpected_keys = self.backbone.load_state_dict(checkpoint, strict=False)
            print(f"缺失的键: {len(missing_keys)}, 意外的键: {len(unexpected_keys)}")
        else:
            print("使用默认 ImageNet 预训练权重")
            self.backbone = dinov3_vitl16(pretrained=True)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # DEM 分支
        self.dem_encoder = DEMResNetExtractor(pretrained=True)

        # DR Fusion Head
        self.decoder = DRFusionHead(img_ch=self.backbone.embed_dim, num_classes=num_classes)

    def forward(self, image, dem):
        # ViT 特征
        img_feat = self.backbone.get_intermediate_layers(
            image, n=1, reshape=True, norm=True
        )[0]  # [B,1024,H/16,W/16]

        # DEM 多尺度
        dem_feats = self.dem_encoder(dem)  # [c1, c2, c3]

        # 解码
        logits = self.decoder(img_feat, dem_feats, image.shape[-2:])
        return logits

    def unfreeze_backbone_layers(self, last_n=8):
        """
        解冻 backbone 最后 n 层 transformer blocks
        """
        total_blocks = len(self.backbone.blocks)
        start_idx = total_blocks - last_n
        for idx, block in enumerate(self.backbone.blocks):
            if idx >= start_idx:
                for p in block.parameters():
                    p.requires_grad = True
        # 解冻最后 LayerNorm
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        print(f" 解冻 backbone 最后 {last_n} 层 Transformer block 和 LayerNorm")


# ----------------------------
# 创建模型
# ----------------------------
def create_model(num_classes, pretrained_path=None, freeze_backbone=True, unfreeze_last_n=8):
    model = DINOv3SegmentationModel(num_classes=num_classes,
                                          pretrained_path=pretrained_path,
                                          freeze_backbone=freeze_backbone)
    if unfreeze_last_n > 0:
        model.unfreeze_backbone_layers(last_n=unfreeze_last_n)
    return model


def create_backbone(pretrained_path=None):
    model = DINOv3SegmentationModel(num_classes=1,
                                          pretrained_path=pretrained_path,
                                          freeze_backbone=True)
    return model.backbone