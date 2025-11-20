# ======================================================
# models_mae.py | Masked Autoencoder (ViT backbone)
# Fully compatible with pretrain_mae.py
# ======================================================

import torch
import torch.nn as nn
from timm import create_model


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder (MAE) implemented with ViT backbone.
    - Uses random masking on input patches
    - Reconstructs all patches (including masked ones)
    - Compatible with CIFAR-100 (32x32)
    """

    def __init__(self, img_size=32, patch_size=4, embed_dim=192, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * 3  # 每个 patch 的像素维度 = 4x4x3=48

        # ------------------------------------------------
        # ViT Encoder (无分类头, 无 pooling)
        # ------------------------------------------------
        self.encoder = create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,
            global_pool=''
        )

        # ------------------------------------------------
        # Decoder: reconstruct each patch
        # 输入 [B, N, D] → 输出 [B, N, patch_dim]
        # ------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.patch_dim)
        )

    # ======================================================
    # Patchify: split image into patches
    # ======================================================
    def patchify(self, imgs):
        """
        imgs: [B, 3, H, W]
        return: [B, num_patches, patch_dim]
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W and H % p == 0, "Image size must be divisible by patch size"
        h = w = H // p
        patches = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B, h * w, p * p * C)
        return patches

    # ======================================================
    # Unpatchify: reconstruct full image from patches
    # ======================================================
    def unpatchify(self, patches):
        """
        patches: [B, num_patches, patch_dim]
        return: [B, 3, H, W]
        """
        p = self.patch_size
        B, N, D = patches.shape
        h = w = int(N ** 0.5)
        assert D == p * p * 3, f"Patch dim mismatch: expected {p*p*3}, got {D}"
        patches = patches.reshape(B, h, w, p, p, 3).permute(0, 5, 1, 3, 2, 4)
        imgs = patches.reshape(B, 3, h * p, w * p)
        return imgs

    # ======================================================
    # Random Masking
    # ======================================================
    def random_masking(self, x):
        """
        Randomly mask patches.
        x: [B, N, D]
        return: x_masked, mask
        """
        N = x.shape[1]
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(x.shape[0], N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([x.shape[0], N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        return x_masked, mask

    # ======================================================
    # Forward: MAE pipeline
    # ======================================================
    def forward(self, imgs):
        """
        imgs: [B, 3, H, W]
        return:
          - pred: reconstructed patches [B, N, patch_dim]
          - mask: [B, N]
          - patches: original patches [B, N, patch_dim]
          - recon_imgs: reconstructed image [B, 3, H, W]
        """
        # 1️⃣ 分块
        patches = self.patchify(imgs)  # [B, N, patch_dim]
        x_masked, mask = self.random_masking(patches)

        # 2️⃣ ViT 编码
        feats = self.encoder.forward_features(imgs)  # [B, N+1, D]
        if feats.shape[1] > self.num_patches:        # 去掉 CLS token
            feats = feats[:, 1:, :]

        # 3️⃣ 解码重建所有 patch
        pred = self.decoder(feats)                   # [B, N, patch_dim]

        # 4️⃣ 组合为完整图像
        recon_imgs = self.unpatchify(pred)
        
        masked_imgs = self.unpatchify(patches * (1 - mask.unsqueeze(-1)))  # 把可视化用的 mask 输入还原
        return pred, mask, patches, recon_imgs, masked_imgs