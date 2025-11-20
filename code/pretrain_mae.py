# ======================================================
# pretrain_mae.py | MAE 自监督预训练流程
# ======================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from models_mae import MaskedAutoencoderViT
from config_data import get_cifar100

def train_mae(train_loader, epochs=100, lr=1e-4,
              save_path='./outputs/checkpoints/mae_vit.pth',
              log_dir='./outputs/tensorboard/mae',
              mask_ratio=0.75):

    # ======================================================
    # 基础设置
    # ======================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = SummaryWriter(log_dir)

    # ======================================================
    # 模型初始化
    # ======================================================
    model = MaskedAutoencoderViT(mask_ratio=mask_ratio).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction='none')

    print(f"\n🚀 Start MAE Pretraining | mask_ratio={mask_ratio} | epochs={epochs}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, _ in tqdm(train_loader, desc=f"MAE Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)

            # 前向传播
            pred, mask, patches, recon_imgs,_ = model(imgs)
            # pred: [B, N, 48]  patches: [B, N, 48]  mask: [B, N]

            # === Masked MSE Loss (仅在mask区域计算) ===
            patch_loss = (mse(pred, patches).mean(dim=-1) * mask).sum() / mask.sum()

            opt.zero_grad()
            patch_loss.backward()
            opt.step()
            total_loss += patch_loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/MAE", avg_loss, epoch)
        print(f"[MAE] Epoch {epoch+1:02d}/{epochs} | Loss={avg_loss:.6f}")

    # ======================================================
    # 训练完成，保存模型
    # ======================================================
    torch.save(model.state_dict(), save_path)
    writer.close()
    print(f"✅ MAE 预训练完成，模型已保存至 {save_path}")


    # ======================================================
    # ✅ 最终保存重建可视化（三行：原图 / Mask 输入 / 重建）
    # ======================================================
    print("\n🧩 正在生成最终重建可视化（原图 + Mask 输入 + 重建）...")
    _, _, clean_loader = get_cifar100(batch_size=8, augment=False)
    clean_imgs, _ = next(iter(clean_loader))
    clean_imgs = clean_imgs.to(device)

    model.eval()
    with torch.no_grad():
        # 获取 masked 输入与重建结果
        pred, mask, patches, recon_imgs, masked_imgs = model(clean_imgs)

        # 拼接：原图 + masked + 重建
        grid = torch.cat([clean_imgs.cpu(), masked_imgs.cpu(), recon_imgs.cpu()], dim=0)
        save_dir = "./outputs/mae_samples"
        os.makedirs(save_dir, exist_ok=True)
        save_path_img = f"{save_dir}/final_reconstruction.png"
        save_image(grid, save_path_img, nrow=8, normalize=True)
        print(f"🖼️ Final 3-row reconstruction saved → {save_path_img}")


    
