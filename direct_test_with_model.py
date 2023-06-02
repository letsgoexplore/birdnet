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
from utils.load_data import *
from utils.load_data_task_g import myDataloader_task_g

# 不同的网络backbone
from nets.resnet50 import *
from nets.vgg16 import *
from nets.diy_net import *
from nets.diy_net_5c_4l import *

# 引入测试模块
from utils.test import mytest


if __name__ == "__main__":
    model_path = 'model/model.pth'
    _, _, _, test_loader = myDataloader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(num_classes=525)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    mytest(device, model, test_loader)
