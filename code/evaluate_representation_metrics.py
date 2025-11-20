# ======================================================
# evaluate_representation_metrics.py | 表示质量定量评估
# ======================================================
import torch, numpy as np, timm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from config_data import get_cifar100
import os

def load_encoder(ptype, path=None):
    if ptype == "Supervised":
        try:
            model = timm.create_model('vit_tiny_patch4_32', pretrained=False,
                                      num_classes=100, global_pool='token')
        except:
            model = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                                      num_classes=100, img_size=32, patch_size=4,
                                      global_pool='token')
        state = torch.load("../CIFAR_project/runs/cifar100_vit_tiny_aug_20251027_223443/best_model_vit_tiny.pth",
                           map_location='cpu')
        model.load_state_dict(state, strict=False)
        return model
    elif ptype == "MAE":
        from models_mae import MaskedAutoencoderViT
        m = MaskedAutoencoderViT(); m.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
        return m.encoder
    elif ptype == "Rotation":
        from models_rotation import RotationPredictionViT
        r = RotationPredictionViT(); r.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
        return r.encoder

    elif ptype == "Random":
        try:
            return timm.create_model('vit_tiny_patch4_32', pretrained=False, num_classes=0, global_pool='token')
        except:
            return timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0,
                                     img_size=32, patch_size=4, global_pool='token')

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
        if f.ndim == 3: f = f.mean(1)
        feats.append(f.cpu())
        labels_all.append(labels)
    return torch.cat(feats), torch.cat(labels_all)

def alignment_uniformity(X):
    """计算对齐性 (alignment) 与均匀性 (uniformity)"""
    X = X / X.norm(dim=1, keepdim=True)
    sim = torch.mm(X, X.t())
    align = (1 - sim.diag().mean()).item()
    uni = np.log(torch.exp(2 * sim).mean()).item()
    return align, uni

def evaluate_representation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_cifar100(batch_size=256, augment=False)

    models = {
        "Random": load_encoder("Random"),
        "Supervised": load_encoder("Supervised"),
        "MAE": load_encoder("MAE", "./outputs/checkpoints/mae_vit.pth"),
        "Rotation": load_encoder("Rotation", "./outputs/checkpoints/rotation_vit.pth"),
    }

    results = []
    for name, model in models.items():
        model.to(device)
        X_train, y_train = extract_features(model, train_loader, device)
        X_test, y_test = extract_features(model, val_loader, device)
        print(f"✨ {name} 特征提取完成: {X_train.shape}")

        # KNN 表示质量评估
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test)) * 100

        # 对齐性 & 均匀性
        align, uni = alignment_uniformity(X_train)

        results.append((name, acc, align, uni))

    print("\n=== 表示质量评估结果 ===")
    for r in results:
        print(f"{r[0]:<12s} | KNN Acc={r[1]:.2f}% | Align={r[2]:.4f} | Uniform={r[3]:.4f}")
    np.savetxt("./outputs/representation_metrics.csv", results, fmt='%s', delimiter=",")
    print("✅ 结果已保存至 ./outputs/representation_metrics.csv")
