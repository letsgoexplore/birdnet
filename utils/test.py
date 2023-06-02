# torch相关
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

# 运行过程中的utils
import pandas as pd
import csv
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import datetime

# 加载全部数据或加载部分数据（项目task g)
from load_data import *
from load_data_task_g import myDataloader_task_g

# 不同的网络backbone
from resnet50 import *
from vgg19 import *
from diy_net import *
from diy_net_5c_4l import *

def mytest(device, model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        total_f1_score = 0
        total_precision = 0
        total_recall = 0
        total_item = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_item += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            labels = labels.cpu().numpy()  # 转换为NumPy数组
            predicted = predicted.cpu().numpy()  # 转换为NumPy数组

            print(labels.shape)
            print(predicted.shape)
            total_f1_score += f1_score(labels, predicted, average='macro')
            total_precision += precision_score(labels, predicted, average='macro')
            total_recall += recall_score(labels, predicted, average='macro')

    accuracy = correct / total
    f1 = total_f1_score / total_item
    precision = total_precision / total_item
    recall = total_recall / total_item

    print(f'Testset Accuracy: {accuracy}')
    print(f'Testset f1 score: {f1}')
    print(f'Testset precision: {precision}')
    print(f'Testset Recall: {recall}')

    return accuracy, f1, precision, recall