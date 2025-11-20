import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models_rotation import RotationPredictionViT
import torchvision.transforms.functional as TF

def random_rotate_batch(x):
    rotated, labels = [], []
    for img in x:
        angle = torch.randint(0, 4, (1,)).item() * 90
        rotated.append(TF.rotate(img, angle))
        labels.append(angle // 90)
    return torch.stack(rotated), torch.tensor(labels)

def train_rotation(train_loader, epochs=200, lr=1e-4, save_path='./outputs/checkpoints/rotation_vit.pth',
                   log_dir='./outputs/tensorboard/rotation'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RotationPredictionViT().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir)

    for e in range(epochs):
        model.train(); total_loss = correct = total = 0
        for imgs, _ in tqdm(train_loader, desc=f"Rotation {e+1}/{epochs}"):
            imgs, labels = random_rotate_batch(imgs)
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = ce(preds, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)
        acc = correct / total * 100
        writer.add_scalar("Loss/Rotation", total_loss / len(train_loader), e)
        writer.add_scalar("Acc/Rotation", acc, e)
        print(f"Epoch {e+1}: Acc={acc:.2f}% Loss={total_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), save_path)
    writer.close()
    print(f"✅ Rotation 预训练完成 → {save_path}")
