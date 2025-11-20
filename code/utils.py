# ======================================================
# utils.py | 通用工具函数合集 (训练 + 评估 + 可视化 + 效率)
# ======================================================

import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from math import log10
from pytorch_msssim import ssim
import timm
import pandas as pd
# ======================================================
# ✅ 基础训练与验证函数
# ======================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, is_test=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    acc = 100. * correct / total
    if is_test:
        print(f"🎯 Test: Loss={total_loss / len(loader.dataset):.4f}, Acc={acc:.2f}%")
    return total_loss / len(loader.dataset), acc


# ======================================================
# ✅ 绘制分类任务训练曲线 (Loss + Accuracy)
# ======================================================
def plot_curves(train_losses, val_losses, val_accuracies, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Training & Validation Loss"); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Val Accuracy", color="orange", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()
    plt.title("Validation Accuracy"); plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 分类曲线图已保存至 {save_path}")
    else:
        plt.show()


# ======================================================
# ✅ 绘制重建任务曲线 (Loss + PSNR + SSIM)
# ======================================================
def plot_reconstruction_metrics(train_losses, psnr_scores, ssim_scores, task_name, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2, color="tab:blue")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.grid(True)
    plt.title(f"{task_name} Training Loss")

    plt.subplot(1, 3, 2)
    plt.plot(epochs, psnr_scores, label="PSNR", linewidth=2, color="tab:green")
    plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)"); plt.grid(True)
    plt.title("Validation PSNR")

    plt.subplot(1, 3, 3)
    plt.plot(epochs, ssim_scores, label="SSIM", linewidth=2, color="tab:red")
    plt.xlabel("Epoch"); plt.ylabel("SSIM"); plt.grid(True)
    plt.title("Validation SSIM")

    plt.tight_layout()
    save_path = f"{output_dir}/{task_name.lower()}_metrics_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 已保存 {task_name} 曲线图 → {save_path}")


# ======================================================
# ✅ 计算图像质量指标 (PSNR + SSIM)
# ======================================================
def compute_metrics(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 20 * log10(1.0 / (mse ** 0.5 + 1e-8))
    ssim_val = ssim(pred.unsqueeze(0), target.unsqueeze(0),
                    data_range=1.0, size_average=True).item()
    return psnr, ssim_val


# ======================================================
# ✅ 计时装饰器
# ======================================================
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ {func.__name__} 总耗时: {(end - start)/60:.2f} min")
        return result
    return wrapper

# ------------------------------------------------------
# ⏱️ 训练时间 & 推理速度
# ------------------------------------------------------
def measure_training_time(func, *args, **kwargs):
    start = time.time()
    output = func(*args, **kwargs)
    end = time.time()
    return output, round((end - start) / 60.0, 2)


def measure_inference_speed(model_path, dataloader, device="cuda", n_images=100):
    """
    测量模型推理速度 (秒 / 100 张图片)
    ✅ 自动识别 state_dict 并重建 ViT 模型结构
    ✅ 使用 torch.cuda.synchronize() 确保计时准确
    """

    if not os.path.exists(model_path):
        print(f"⚠️ 未找到模型权重: {model_path}")
        return None

    state = torch.load(model_path, map_location=device)

    # ✅ 若是 state_dict，则自动重建 ViT Tiny 模型
    if isinstance(state, dict):
        print(f"🔧 检测到 state_dict，自动构建 ViT 模型用于推理测速...")
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            img_size=32, patch_size=4,
            num_classes=100
        ).to(device)
        model.load_state_dict(state, strict=False)
    else:
        model = state.to(device)

    model.eval()

    imgs_list = []
    for imgs, _ in dataloader:
        imgs_list.append(imgs)
        if sum([b.size(0) for b in imgs_list]) >= n_images:
            break
    imgs = torch.cat(imgs_list)[:n_images].to(device)

    # ✅ 同步 CUDA 流，确保时间测量准确
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()

    with torch.no_grad():
        _ = model(imgs)

    torch.cuda.synchronize() if device == "cuda" else None
    end = time.time()

    elapsed = round(end - start, 3)
    print(f"⚡ 推理时间: {elapsed:.3f} s / 100 images ({os.path.basename(model_path)})")

    return elapsed

# ------------------------------------------------------
# 📊 绘制 数据规模 vs 精度 曲线
# ------------------------------------------------------
def plot_data_scale(csv_path="./outputs/data_scale.csv",
                    save_path="./outputs/plot_data_scale.png"):
    if not os.path.exists(csv_path):
        print(f"❌ 未找到 {csv_path}")
        return
    df = pd.read_csv(csv_path).sort_values("DataRatio")
    plt.figure(figsize=(6, 4))
    plt.plot(df["DataRatio"], df["Accuracy(%)"], marker="o", linewidth=2.2)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title("MAE Pretraining Data Scale vs Classification Accuracy")
    plt.xlabel("Training Data Ratio"); plt.ylabel("Accuracy (%)")
    plt.xticks(df["DataRatio"], [f"{r*100:.0f}%" for r in df["DataRatio"]])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 数据规模曲线已保存至 {save_path}")

'''
# ======================================================
# ✅ 保存效率指标 (JSON)
# ======================================================
def record_efficiency(train_time_min, inference_time_s, acc_gain,
                      save_path="./outputs/efficiency.json"):
    result = {
        "Training Time (min)": round(train_time_min, 2),
        "Inference Time (s / 100 images)": round(inference_time_s, 3),
        "Accuracy Gain (%)": round(acc_gain, 2)
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"✅ 训练效率结果已保存至 {save_path}")
'''