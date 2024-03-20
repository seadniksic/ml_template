import sys
sys.path.append("..")

import os

import seaborn
import framework.Graphing as g
import framework.LogConversion as l
import matplotlib.pyplot as plt

RUN_NAME = "20240312-213006"
LOG_BASE_PATH = os.path.join("../../output/train_output/ResNetBinary/runs", RUN_NAME)
LOG_TRAIN_PATH =  os.path.join(LOG_BASE_PATH, "train_losses.txt")
LOG_VALIDATION_PATH = os.path.join(LOG_BASE_PATH, "validation_losses.txt")

train_data = l.get_2D_loss_data(LOG_TRAIN_PATH)
validation_data = l.get_2D_loss_data(LOG_VALIDATION_PATH)

grapher = g.Graphing()

train_plt = grapher.line_plot_2D(*train_data, "batch number", "loss")
train_plt.figure.savefig(os.path.join(LOG_BASE_PATH, "train_graph.png"))

plt.clf()

validation_plt = grapher.line_plot_2D(*validation_data, "batch number", "loss")
validation_plt.figure.savefig(os.path.join(LOG_BASE_PATH, "validation_graph.png"))




