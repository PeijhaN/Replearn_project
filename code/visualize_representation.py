'''
# ======================================================
# visualize_representation.py | t-SNE & UMAP 表示可视化 (可自定义类别数)
# ======================================================

import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from config_data import get_cifar100
import itertools

# ------------------------------------------------------
# 加载模型
# ------------------------------------------------------
def load_vit_supervised(path, num_classes=100):
    """加载实验一的有监督 ViT 模型"""
    try:
        model = timm.create_model('vit_tiny_patch4_32', pretrained=False,
                                  num_classes=num_classes, global_pool='token')
    except:
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                                  num_classes=num_classes, img_size=32, patch_size=4,
                                  global_pool='token')
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    print(f"✅ 已加载监督模型: {path}")
    return model


def load_mae_encoder(path):
    from models_mae import MaskedAutoencoderViT
    mae = MaskedAutoencoderViT()
    state = torch.load(path, map_location='cpu')
    mae.load_state_dict(state, strict=False)
    print(f"✅ 已加载 MAE 模型: {path}")
    return mae.encoder


def load_rotation_encoder(path):
    from models_rotation import RotationPredictionViT
    rot = RotationPredictionViT()
    state = torch.load(path, map_location='cpu')
    rot.load_state_dict(state, strict=False)
    print(f"✅ 已加载 Rotation 模型: {path}")
    return rot.encoder


# ------------------------------------------------------
# 特征提取
# ------------------------------------------------------
@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labels_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        try:
            f = model.forward_features(imgs)
        except:
            f = model(imgs)
        if isinstance(f, (list, tuple)):
            f = f[0]
        if f.ndim == 3:
            f = f.mean(1)
        feats.append(f.cpu())
        labels_all.append(labels)
    return torch.cat(feats), torch.cat(labels_all)


# ------------------------------------------------------
# 可视化主函数
# ------------------------------------------------------
def visualize_representation(method="tsne",
                             save_path="./outputs/representation_tsne_umap.png",
                             num_classes_to_show=20):
    """
    通用表示可视化函数 (t-SNE / UMAP)
    num_classes_to_show: 仅展示前 N 个类别
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_cifar100(batch_size=256, augment=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = {
        "Random Init": timm.create_model('vit_tiny_patch16_224',
                                         pretrained=False, num_classes=0,
                                         img_size=32, patch_size=4, global_pool='token'),
        "Supervised ViT": load_vit_supervised(
            "../CIFAR_project/runs/cifar100_vit_tiny_aug_20251027_223443/best_model_vit_tiny.pth"),
        "MAE": load_mae_encoder("./outputs/checkpoints/mae_vit.pth"),
        "Rotation": load_rotation_encoder("./outputs/checkpoints/rotation_vit.pth")
    }

    feats_all, labels_all, names = [], [], []
    for name, model in models.items():
        model.to(device)
        feats, labels = extract_features(model, test_loader, device)
        feats_all.append(feats)
        labels_all.append(labels)
        names.append(name)
        print(f"✨ 提取特征完成: {name}, shape={feats.shape}")

    # ====== 降维 & 绘图 ======
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = np.ravel(axes)
    palette = np.array(sns.color_palette("husl", num_classes_to_show))
    

    for i, (X, y, name) in enumerate(zip(feats_all, labels_all, names)):
        mask = (y < num_classes_to_show)
        X, y = X[mask], y[mask]

        idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
        X_sub, y_sub = X[idx], y[idx]

        if method == "tsne":
            reducer = TSNE(n_components=2, perplexity=50, random_state=42)
        else:
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
        X_2d = reducer.fit_transform(X_sub)

        colors = palette[y_sub.numpy() % num_classes_to_show]
        axes[i].scatter(X_2d[:, 0], X_2d[:, 1],
                        color=colors, s=12, alpha=0.9, edgecolors='none')
        axes[i].set_title(name, fontsize=12)
        axes[i].axis("off")

    # ====== 图例
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=palette[j], label=f"{j}")
               for j in range(num_classes_to_show)]
    
    fig.legend(handles=handles,
               loc="center right",
               bbox_to_anchor=(1.05, 0.5),
               fontsize=7,
               ncol=2,
               frameon=False,
               title=f"CIFAR-100 ")

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ {method.upper()} 可视化图已保存至 {save_path}")

    

# ------------------------------------------------------
# 主入口：生成 t-SNE 与 UMAP 各一张
# ------------------------------------------------------
def visualize_tsne_umap(num_classes_to_show=100):
    visualize_representation(method="tsne",
                             save_path=f"./outputs/representation_tsne_top{num_classes_to_show}.png",
                             num_classes_to_show=num_classes_to_show)
    visualize_representation(method="umap",
                             save_path=f"./outputs/representation_umap_top{num_classes_to_show}.png",
                             num_classes_to_show=num_classes_to_show)

 '''
import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from matplotlib import colors as mcolors
from colorspacious import cspace_convert
from config_data import get_cifar100
import colorsys

import numpy as np
import colorsys
from colorspacious import cspace_convert

def generate_vivid_distinct_colors(n=50, seed=42):
    """
    生成 n 个肉眼可分的高亮彩色 (混合黄金比例 + Lab 感知均匀)
    返回: [(r,g,b), ...] ∈ [0,1]
    """
    np.random.seed(seed)
    colors = []
    golden_ratio = 0.618033988749895
    hue = np.random.rand()

    # Step 1: 先生成 n 个 hue（打散色相）
    hues = [(hue + i * golden_ratio) % 1 for i in range(n)]

    # Step 2: 转换到 Lab 空间调整亮度与饱和度
    for i, h in enumerate(hues):
        l = 70 + 10 * np.sin(i * 0.5)    # 亮度在 60~80 之间波动（亮色）
        s = 0.85                         # 高饱和度
        rgb = colorsys.hls_to_rgb(h, l / 100.0, s)

        # 转换到 Lab 空间确保视觉均匀
        lab = cspace_convert(np.array(rgb)[None, :], "sRGB1", "CIELab")
        lab[0, 0] = np.clip(lab[0, 0] + np.random.uniform(-5, 5), 40, 80)
        rgb = cspace_convert(lab, "CIELab", "sRGB1")[0]
        rgb = np.clip(rgb, 0, 1)
        colors.append(tuple(rgb))

    return np.array(colors)


# ======================================================
# 特征提取
# ======================================================
@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labels_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        try:
            f = model.forward_features(imgs)
        except:
            f = model(imgs)
        if isinstance(f, (list, tuple)):
            f = f[0]
        if f.ndim == 3:
            f = f.mean(1)
        feats.append(f.cpu())
        labels_all.append(labels)
    return torch.cat(feats), torch.cat(labels_all)


# ======================================================
# 模型加载
# ======================================================
def load_vit_supervised(path, num_classes=100):
    try:
        model = timm.create_model('vit_tiny_patch4_32', pretrained=False,
                                  num_classes=num_classes, global_pool='token')
    except:
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                                  num_classes=num_classes, img_size=32, patch_size=4,
                                  global_pool='token')
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    print(f"✅ 已加载监督模型: {path}")
    return model


def load_mae_encoder(path):
    from models_mae import MaskedAutoencoderViT
    mae = MaskedAutoencoderViT()
    mae.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    print(f"✅ 已加载 MAE 模型: {path}")
    return mae.encoder


def load_rotation_encoder(path):
    from models_rotation import RotationPredictionViT
    rot = RotationPredictionViT()
    rot.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    print(f"✅ 已加载 Rotation 模型: {path}")
    return rot.encoder


# ======================================================
# 可视化函数
# ======================================================
def visualize_representation(method="tsne",
                             save_path="./outputs/representation_tsne_lab_color.png",
                             num_classes_to_show=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_cifar100(batch_size=256, augment=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = {
        "Random Init": timm.create_model('vit_tiny_patch16_224',
                                         pretrained=False, num_classes=0,
                                         img_size=32, patch_size=4, global_pool='token'),
        "Supervised ViT": load_vit_supervised(
            "../CIFAR_project/runs/cifar100_vit_tiny_aug_20251027_223443/best_model_vit_tiny.pth"),
        "MAE": load_mae_encoder("./outputs/checkpoints/mae_vit.pth"),
        "Rotation": load_rotation_encoder("./outputs/checkpoints/rotation_vit.pth")
    }

    feats_all, labels_all, names = [], [], []
    for name, model in models.items():
        model.to(device)
        feats, labels = extract_features(model, test_loader, device)
        feats_all.append(feats)
        labels_all.append(labels)
        names.append(name)
        print(f"✨ 提取特征完成: {name}, shape={feats.shape}")

    # ====== 降维 & 绘图 ======
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = np.ravel(axes)

    #colors_unique = generate_perceptually_distinct_colors(50)

    colors_unique = generate_vivid_distinct_colors(50)

    markers = ['o', '^']  # 圆形 + 五边形

    for i, (X, y, name) in enumerate(zip(feats_all, labels_all, names)):
        mask = (y < num_classes_to_show)
        X, y = X[mask], y[mask]
        idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
        X_sub, y_sub = X[idx], y[idx]

        reducer = (TSNE(n_components=2, perplexity=50, random_state=42)
                   if method == "tsne"
                   else umap.UMAP(n_neighbors=30, min_dist=0.1,
                                  metric='cosine', random_state=42))
        X_2d = reducer.fit_transform(X_sub)

        for cls in range(num_classes_to_show):
            idx_cls = (y_sub == cls)
            color = colors_unique[cls % 50]  # 每形状组复用
            marker = markers[cls // 50]
            axes[i].scatter(X_2d[idx_cls, 0], X_2d[idx_cls, 1],
                            color=[color], marker=marker,
                            s=16, alpha=0.8, linewidths=0)

        axes[i].set_title(name, fontsize=12)
        axes[i].axis("off")

    # ====== 图例 ======
    handles = []
    for cls in range(num_classes_to_show):
        color = colors_unique[cls % 50]
        marker = markers[cls // 50]
        handles.append(plt.Line2D([], [], marker=marker, linestyle="",
                                  color=color, label=f"{cls}", markersize=6))
    fig.legend(handles=handles, loc="center right",
               bbox_to_anchor=(1.1, 0.5),
               fontsize=7, ncol=2, frameon=False,
               title="CIFAR-100 Classes")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {method.upper()} 可视化图已保存至 {save_path}")


# ======================================================
# 主入口
# ======================================================
def visualize_tsne_umap(num_classes_to_show=100):
    visualize_representation(method="tsne",
                             save_path=f"./outputs/representation_tsne_lab_color.png",
                             num_classes_to_show=num_classes_to_show)
    visualize_representation(method="umap",
                             save_path=f"./outputs/representation_umap_lab_color.png",
                             num_classes_to_show=num_classes_to_show)
