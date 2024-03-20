import sys
sys.path.append('..')

import torch
from datasets.Dataset import Dataset
from models.InceptionBinary import InceptionBinary
from framework.SupervisedMLFramework import SupervisedMLFramework
from losses.inception_binary_loss import InceptionBinaryLoss
from src.transforms.PytorchPretrainedTransform import PytorchPretrainedTransform

TEST_IMAGE_PATH = "../../../../../data/new_data/prepared/test/images"
TEST_LABEL_PATH = "../../../../../data/new_data/prepared/test/labels/labels.pkl"

test_dataset = OEDDataset(TEST_IMAGE_PATH, TEST_LABEL_PATH, transform=PytorchPretrainedTransform())
model = InceptionBinary()
model.load_state_dict(...)

loss_function = InceptionBinaryLoss()

framework = SupervisedMLFramework(model, "InceptionBinary", "../output/test_output", None, test_dataset)
framework.test(loss_function, batch_size=4)
