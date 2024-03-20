import sys
sys.path.append('..')

import torch
from datasets.OEDDataset import OEDDataset
from models.InceptionBinary import InceptionBinary
from framework.SupervisedMLFramework import SupervisedMLFramework
from losses.inception_binary_loss import InceptionBinaryLoss
from transforms.PytorchPretrainedTransform import PytorchPretrainedTransform
from models.ResNetBinary import ResNetBinary

TRAIN_IMAGE_PATH = "../../../../../data/train/images"
TRAIN_LABEL_PATH = "../../../../../data/train/labels/labels.pkl"

INPUT_SIZE = 512
USE_CUSTOM_VALIDATION = True

if USE_CUSTOM_VALIDATION:
    VALIDATION_IMAGE_PATH = "../../../../../data/validation/images"
    VALIDATION_LABEL_PATH = "../../../../../data/validation/labels/labels.pkl"
    validation_dataset = OEDDataset(VALIDATION_IMAGE_PATH, VALIDATION_LABEL_PATH, transform=PytorchPretrainedTransform())

train_dataset = OEDDataset(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, transform=PytorchPretrainedTransform())

model = ResNetBinary(pretrained=True)

""" --- Hyper Parameters --- """
lr = .001
epochs = 75
weight_decay = .01

loss_function = InceptionBinaryLoss()
optim = torch.optim.Adam
optim_params = {"lr": lr, "weight_decay": weight_decay}
scheduler = torch.optim.lr_scheduler.StepLR
sched_params={"step_size": 25}
framework = SupervisedMLFramework(model, "ResNetBinary", "../output/train_output", train_dataset, custom_validation_dataset= validation_dataset if USE_CUSTOM_VALIDATION else None)
framework.train(epochs, loss_function, optim, optim_params, sched=scheduler, sched_params=sched_params, batch_size=32, weight_save_period=5, patience=10, use_custom_validation_set=USE_CUSTOM_VALIDATION, validation_percent=20)
