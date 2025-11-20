import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from timm import create_model

class RotationPredictionViT(nn.Module):
    """
    Rotation 预测任务：与 MAE encoder 结构一致
    """
    def __init__(self, embed_dim=192, num_classes=4):
        super().__init__()
        self.encoder = create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            img_size=32,
            patch_size=4,
            num_classes=0,
            global_pool=''
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, labels=None):
        feats = self.encoder.forward_features(x)
        feats = feats.mean(dim=1)
        logits = self.head(feats)
        return logits
