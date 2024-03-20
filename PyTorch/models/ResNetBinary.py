import torch
from torchvision import models

class ResNetBinary(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        self.resnet = models.resnet50(weights='IMAGENET1K_V2' if self.pretrained else None)
        self.resnet.fc = torch.nn.Linear(2048, 2)

    def reset(self):
        self.resnet = models.resnet50(weights='IMAGENET1K_V2' if self.pretrained else None)
        self.resnet.fc = torch.nn.Linear(2048, 2)

    def forward(self, x):
        out = self.resnet(x)
        return out


if __name__ == "__main__":
    t = ResNetBinary(pretrained=True)