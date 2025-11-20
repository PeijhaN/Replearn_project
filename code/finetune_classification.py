# ======================================================
# finetune_classification.py | 兼容 MAE / Rotation / Random Init 微调
# ======================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import timm
from utils import plot_curves  # ✅ 引用工具函数


# ======================================================
# 通用分类头：适配 MAE / Rotation 表示
# ======================================================
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim=192, num_classes=100, pool='mean'):
        super().__init__()
        self.pool = pool
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, feats):
        # feats: [B, N, D] 或 [B, D]
        if feats.ndim == 3:
            if self.pool == 'token':
                feats = feats[:, 0]  # CLS token
            else:
                feats = feats.mean(dim=1)  # 平均所有 patch
        return self.fc(feats)


# ======================================================
# 自动检测预训练类型
# ======================================================
def detect_pretrain_type(path):
    if path is None:
        return "Random"
    if "mae" in path.lower():
        return "MAE"
    if "rotation" in path.lower():
        return "Rotation"
    return "Unknown"


# ======================================================
# 通用 ViT 微调函数
# ======================================================
def finetune_vit(train_loader, val_loader, epochs=50, lr=1e-4,
                 pretrain_path=None,
                 save_path='./outputs/checkpoints/ft_MAE.pth',
                 log_dir='./outputs/tensorboard/finetune',
                 num_classes=100,
                 patience=10):
    """
    通用微调接口：
    - 自动识别预训练模型 (MAE / Rotation / Random)
    - 自动调整 pooling 策略
    - 加入 Early Stopping
    - 输出 TensorBoard & 曲线图
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ptype = detect_pretrain_type(pretrain_path)
    print(f"\n🔍 检测到预训练类型: {ptype}")
    #save_path=f'./outputs/checkpoints/ft_{ptype}.pth'
    
    # ======================================================
    # 1️⃣ 根据类型加载 encoder
    # ======================================================
    if ptype == "MAE":
        from models_mae import MaskedAutoencoderViT
        encoder = MaskedAutoencoderViT().encoder.to(device)
        pool = "mean"
    else:
        encoder = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            img_size=32,
            patch_size=4,
            num_classes=0,
            global_pool='token'
        ).to(device)
        pool = "token"

    # ======================================================
    # 2️⃣ 加载预训练权重（仅 encoder 部分）
    # ======================================================
    if pretrain_path and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location=device)
        if isinstance(state, dict) and 'encoder' in state:
            encoder.load_state_dict(state['encoder'], strict=False)
        else:
            encoder.load_state_dict(state, strict=False)
        print(f"✅ 加载 encoder 权重自 {pretrain_path}")

    print(f"📎 Pooling 策略: {pool}")

    # ======================================================
    # 3️⃣ 构建分类模型
    # ======================================================
    head = ClassificationHead(embed_dim=192, num_classes=num_classes, pool=pool).to(device)
    model = nn.Sequential(encoder, head)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    best_acc = 0.0
    train_losses, val_losses, val_accuracies = [], [], []
    no_improve = 0

    # ======================================================
    # 4️⃣ 微调训练循环
    # ======================================================
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"[{ptype}] Fine-tune Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            feats = model[0].forward_features(imgs)
            logits = model[1](feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # ===== 验证阶段 =====
        model.eval()
        val_total_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = model[0].forward_features(imgs)
                logits = model[1](feats)
                loss = criterion(logits, labels)
                val_total_loss += loss.item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_total_loss / len(val_loader)
        val_acc = val_correct / val_total * 100
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        writer.add_scalar(f"Loss/Train_{ptype}", train_loss, epoch)
        writer.add_scalar(f"Loss/Val_{ptype}", val_loss, epoch)
        writer.add_scalar(f"Accuracy/Val_{ptype}", val_acc, epoch)

        print(f"[{ptype}] Epoch {epoch+1}: TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | ValAcc={val_acc:.2f}%")

        # ===== 早停机制 =====
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⏹️ Early Stopping triggered at epoch {epoch+1}")
                break
            
    os.makedirs('./outputs/ft_classification',exist_ok = True)
    writer.close() 
    
    plot_save_path = os.path.join('./outputs/ft_classification', os.path.basename(save_path))
    
    plot_save_path = plot_save_path.replace( '.pth', '_curve.png')
    plot_curves(train_losses, val_losses, val_accuracies, save_path=plot_save_path)
    print(f"✅ Finetune ({ptype}) 完成 | 最佳精度: {best_acc:.2f}% | 模型已保存至 {save_path}")

    return best_acc
