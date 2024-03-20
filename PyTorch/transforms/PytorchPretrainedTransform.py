import torch
from torchvision import transforms
from PIL import Image

class PytorchPretrainedTransform(torch.nn.Module):

    def forward(self, img):

        img = Image.fromarray(img, mode="RGB")

        custom_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        new_img = custom_transform(img)

        return new_img
