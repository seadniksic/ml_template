import torch

class InceptionBinaryLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, x, y): 
        return self.loss_function(x, y) 

