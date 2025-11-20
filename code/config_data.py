import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

def get_cifar100(batch_size=128, subset_ratio=1.0, augment=True):
    """
    加载 CIFAR-100 数据集（统一增强版本）
    支持 subset_ratio 控制预训练数据规模。
    """
    normalize = transforms.Normalize((0.5071, 0.4865, 0.4409),
                                     (0.2673, 0.2564, 0.2762))

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # ✅ 加载数据
    train_full = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    # ✅ 子集控制（用于数据规模实验）
    if subset_ratio < 1.0:
        subset_len = int(len(train_full) * subset_ratio)
        idx = np.random.choice(len(train_full), subset_len, replace=False)
        train_full = Subset(train_full, idx)
        print(f"📦 使用 {subset_len} / {len(train_full.dataset)} ({subset_ratio*100:.1f}%) 样本进行训练")

    # ✅ 训练/验证划分
    train_size = int(0.9 * len(train_full))
    val_size = len(train_full) - train_size
    train_set, val_set = random_split(train_full, [train_size, val_size])

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2),
    )
