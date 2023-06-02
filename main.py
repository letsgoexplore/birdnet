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
    # step1: 设置基本参数
    train_batch = 32
    val_batch = 256
    test_batch = 32
    num_epochs = 5
    lr = 0.001
    batch_interval_to_val = 100 # 每隔100个batch进行一次测试
    early_stop_flag = 10 #连续10次validation的accuracy下降
    num_classes = 250

    # step2: 初始化各模块
    if (num_classes==525):
        name_dic, train_loader, val_loader, test_loader = myDataloader(train_batch, val_batch, test_batch)
    else:
        name_dic, train_loader, val_loader, test_loader = myDataloader_task_g(train_batch, val_batch, test_batch, num_classes)
    model = resnet50(num_classes=num_classes)
    model_name = 'resnet50'
    # model = DIYNet_5c4l(num_classes=num_classes)
    # model_name = 'diy_net_5c4l'
    # model = vgg19(num_classes=num_classes)
    # model_name = 'vgg19'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    
    # step3：准备记录
    # 创建TensorBoard的SummaryWriter对象
    currentTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    writer = SummaryWriter(log_dir=f'./logs/' + currentTime + '' + model_name + '' + str(num_classes))
    writer_num = 0 #第几次记录到writer中

    # step4： 开始训练
    previous_accuracy = 0 # 上一次的accuracy
    decrease_num = 0 # 已经连续下降了多少次
    for epoch in range(num_epochs):
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        model.train()
        train_counter = 0 # 记录当前的是第几个batch
        for images, labels in train_loader:
            # 开始训练
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_counter += 1
            if(train_counter%batch_interval_to_val == 0):
                model.eval()
                with torch.no_grad():
                    # correct = 0
                    # total = 0
                    for images, labels in val_loader: # 只执行一个循环
                        images = images.to(device)
                        labels = labels.to(device)
                        break
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    # total += labels.size(0)
                    # correct += (predicted == labels).sum().item()        
                    accuracy = (predicted == labels).sum().item() / labels.size(0)

                    # 记录这一周期的Loss和Accuracy
                    writer_num += 1
                    writer.add_scalar('Loss/valid', loss.item(), writer_num)
                    writer.add_scalar('Accuracy/valid', accuracy, writer_num)
                    
                    # 早停判断
                    if (accuracy < previous_accuracy):
                        decrease_num += 1
                    else:
                        decrease_num = 0
                    if (decrease_num == early_stop_flag):
                        break
                    previous_accuracy = accuracy
                    print(f'Val_num: {writer_num}, Accuracy: {accuracy}')
                model.train() # 重新恢复为训练模式

            # 更新进度条
            train_bar.set_postfix(loss=loss.item())
            train_bar.update()
        train_bar.close()
        torch.cuda.empty_cache()

    # 保存模型（防止测试过程出问题）
    torch.save(model.state_dict(), f'model/model_{model_name}_{num_classes}_{currentTime}.pth')

    # step5：开始测试
    accuracy, f1, precision, recall =  mytest(device, model, test_loader) 
    
    # 将训练结果写入TensorBoard
    writer.add_scalar('accuracy/test', accuracy)
    writer.add_scalar('f1 score/test', f1)
    writer.add_scalar('precision/test', precision)
    writer.add_scalar('recall/test', recall)
    writer.close()
    
    