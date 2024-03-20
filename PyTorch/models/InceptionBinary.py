import torch
from torchvision import models

class InceptionBinary(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        self.inception = models.inception_v3(weights='DEFAULT' if self.pretrained else None, init_weights=(not self.pretrained))
        self.inception.fc = torch.nn.Linear(2048, 2)

    def reset(self):
        self.inception = models.inception_v3(weights='DEFAULT' if self.pretrained else None, init_weights=(not self.pretrained))        
        self.inception.fc = torch.nn.Linear(2048, 2)

    def forward(self, x):
        out = self.inception(x)
        return out.logits if self.training else out
