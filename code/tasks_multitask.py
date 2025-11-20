# ======================================================
# tasks_multitask.py | 下游任务微调 (上色 + 去噪 + 质量指标 + 可视化)
# ======================================================

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import timm
from finetune_classification import detect_pretrain_type
from utils import compute_metrics, plot_reconstruction_metrics
from config_data import get_cifar100


# ======================================================
# 上色与去噪解码器
# ======================================================
class ColorizationHead(nn.Module):
    def __init__(self, embed_dim=192):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feats_2d):
        return self.decoder(feats_2d)


class DenoisingHead(nn.Module):
    def __init__(self, embed_dim=192):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feats_2d):
        return self.decoder(feats_2d)


# ======================================================
# 加载预训练 Encoder
# ======================================================
def load_pretrained_encoder(pretrain_path, device):
    ptype = detect_pretrain_type(pretrain_path)
    if ptype == "MAE":
        from models_mae import MaskedAutoencoderViT
        encoder = MaskedAutoencoderViT().encoder.to(device)
    else:
        encoder = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            img_size=32,
            patch_size=4,
            num_classes=0,
            global_pool='token'
        ).to(device)

    if pretrain_path and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location=device)
        if isinstance(state, dict) and 'encoder' in state:
            encoder.load_state_dict(state['encoder'], strict=False)
        else:
            encoder.load_state_dict(state, strict=False)
        print(f"✅ Encoder 权重加载自 {pretrain_path}")
    return encoder, ptype


# ======================================================
# 通用重建任务训练函数（上色/去噪）
# ======================================================
def train_reconstruction_task(task_name, encoder, decoder,
                              train_loader, val_loader,
                              output_dir, device, epochs=80):
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(output_dir)
    writer_log_dir = f'./outputs/tensorboard/{base}'

    writer = SummaryWriter(log_dir = writer_log_dir)

    #writer = SummaryWriter(log_dir=f"{output_dir}/tensorboard_{task_name}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    train_losses, psnr_scores, ssim_scores = [], [], []

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.0

        for imgs, _ in tqdm(train_loader, desc=f"[{task_name}] Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            # === 输入构造 ===
            if task_name == "Colorization":
                inputs = imgs.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            elif task_name == "Denoising":
                noise = torch.randn_like(imgs) * 0.2
                inputs = torch.clamp(imgs + noise, 0, 1)
            else:
                raise ValueError("未知任务类型")

            # === 前向传播 ===
            feats = encoder.forward_features(inputs)
            B, N, D = feats.shape
            if N > 64:
                feats = feats[:, 1:, :]  # 去除 CLS token
            h = w = int(feats.shape[1] ** 0.5)
            feats_2d = feats.permute(0, 2, 1).reshape(B, D, h, w)
            preds = decoder(feats_2d)

            loss = criterion(preds, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        writer.add_scalar("Loss/Train", avg_loss, epoch)

        # ===== 验证阶段 =====
        encoder.eval(); decoder.eval()
        psnr_epoch, ssim_epoch, count = 0, 0, 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                if task_name == "Colorization":
                    inputs = imgs.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                else:
                    noise = torch.randn_like(imgs) * 0.2
                    inputs = torch.clamp(imgs + noise, 0, 1)
                feats = encoder.forward_features(inputs)
                B, N, D = feats.shape
                if N > 64:
                    feats = feats[:, 1:, :]
                h = w = int(feats.shape[1] ** 0.5)
                feats_2d = feats.permute(0, 2, 1).reshape(B, D, h, w)
                preds = decoder(feats_2d)
                for i in range(min(8, imgs.size(0))):
                    ps, ss = compute_metrics(preds[i].cpu(), imgs[i].cpu())
                    psnr_epoch += ps
                    ssim_epoch += ss
                    count += 1
        psnr_epoch /= count
        ssim_epoch /= count
        psnr_scores.append(psnr_epoch)
        ssim_scores.append(ssim_epoch)
        writer.add_scalar("PSNR/Val", psnr_epoch, epoch)
        writer.add_scalar("SSIM/Val", ssim_epoch, epoch)

        print(f"📊 {task_name} | Epoch {epoch+1}: Loss={avg_loss:.4f}, PSNR={psnr_epoch:.2f}, SSIM={ssim_epoch:.4f}")

    writer.close()
    plot_reconstruction_metrics(train_losses, psnr_scores, ssim_scores, task_name, output_dir)

    
    # ✅ 保存样例图像对比（使用未增强数据）
  
    with torch.no_grad():
        # 加载无增强版本的数据
        _, _, clean_loader = get_cifar100(batch_size=8, augment=False)
        clean_imgs, _ = next(iter(clean_loader))
        clean_imgs = clean_imgs.to(device)

        # 构造输入（灰度图或加噪声）
        if task_name == "Colorization":
            inputs = clean_imgs.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        elif task_name == "Denoising":
            noise = torch.randn_like(clean_imgs) * 0.2
            inputs = torch.clamp(clean_imgs + noise, 0, 1)

        # 模型推理
        feats = encoder.forward_features(inputs)
        B, N, D = feats.shape
        if N > 64:
            feats = feats[:, 1:, :]
        h = w = int(feats.shape[1] ** 0.5)
        feats_2d = feats.permute(0, 2, 1).reshape(B, D, h, w)
        preds = decoder(feats_2d)

        # 拼接对比：输入 / 输出 / 原图
        grid = torch.cat([inputs.cpu(), preds.cpu(), clean_imgs.cpu()], dim=0)
        save_image(grid, f"{output_dir}/{task_name.lower()}_comparison.png", nrow=8)
        print(f"✅ {task_name} 样例已保存至 {output_dir}（使用未增强数据）")

        


        # ✅ 保存最终指标 JSON
        metrics = {
            "Final Train Loss": round(train_losses[-1], 4),
            "Final PSNR": round(psnr_scores[-1], 3),
            "Final SSIM": round(ssim_scores[-1], 4)
        }
        with open(f"{output_dir}/{task_name.lower()}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"📄 {task_name} 指标已保存 → {output_dir}/{task_name.lower()}_metrics.json")


# ======================================================
# 主函数：上色与去噪独立微调
# ======================================================
def run_multitask_finetune(pretrain_path=None, output_dir="./outputs", epochs=80):
    from config_data import get_cifar100
    train_loader, val_loader, _ = get_cifar100(batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_color, ptype = load_pretrained_encoder(pretrain_path, device)
    print(f"\n🎨 启动上色任务微调 ({ptype})")
    color_head = ColorizationHead().to(device)
    train_reconstruction_task("Colorization", encoder_color, color_head,
                              train_loader, val_loader,
                              f"{output_dir}/{ptype}_Colorization",
                              device=device, epochs=epochs)

    encoder_denoise, _ = load_pretrained_encoder(pretrain_path, device)
    print(f"\n🧹 启动去噪任务微调 ({ptype})")
    denoise_head = DenoisingHead().to(device)
    train_reconstruction_task("Denoising", encoder_denoise, denoise_head,
                              train_loader, val_loader,
                              f"{output_dir}/{ptype}_Denoising",
                              device=device, epochs=epochs)

    print(f"\n✅ {ptype} 下游任务（上色 + 去噪）全部完成 | 输出路径: {output_dir}")
