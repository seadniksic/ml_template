import sys
sys.path.append('..')

import torch
from datasets.OEDDataset import OEDDataset
from models.InceptionBinary import InceptionBinary
from framework.SupervisedMLFramework import SupervisedMLFramework
from losses.inception_binary_loss import InceptionBinaryLoss
from src.transforms.PytorchPretrainedTransform import PytorchPretrainedTransform

TRAIN_IMAGE_PATH = "../../../../../data/new_data/prepared/train/images"
TRAIN_LABEL_PATH = "../../../../../data/new_data/prepared/train/labels/labels.pkl"

USE_CUSTOM_VALIDATION = True

if USE_CUSTOM_VALIDATION:
    VALIDATION_IMAGE_PATH = "../../../../../data/new_data/prepared/validation/images"
    VALIDATION_LABEL_PATH = "../../../../../data/new_data/prepared/validation/labels/labels.pkl"
    validation_dataset = OEDDataset(VALIDATION_IMAGE_PATH, VALIDATION_LABEL_PATH, transform=PytorchPretrainedTransform())

INPUT_SIZE = 512

train_dataset = OEDDataset(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, transform=PytorchPretrainedTransform())
model = InceptionBinary()

""" --- Hyper Parameters --- """
lr = .001
epochs = 20
weight_decay = .01

loss_function = InceptionBinaryLoss()
optim = torch.optim.Adam
optim_params = {"lr": lr, "weight_decay": weight_decay}
scheduler = torch.optim.lr_scheduler.StepLR
sched_params={"step_size": 25}
framework = SupervisedMLFramework(model, "InceptionBinary", "../output/tune_output", train_dataset, custom_validation_dataset= validation_dataset if USE_CUSTOM_VALIDATION else None)
framework.tune(epochs, loss_function, optim, optim_params, sched=scheduler, sched_params=sched_params, batch_size=32, k=5, use_custom_validation_set=USE_CUSTOM_VALIDATION)











