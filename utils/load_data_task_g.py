import pandas as pd
import csv
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image_original = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image_original)
        return image, label

def myDataloader_task_g(train_batch, val_batch, test_batch, class_num):
    # step 0: 设置参数
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # step 1: 将labels转换为数字
    # row[0]是id，row[1]是路径
    # row[2]是labels，row[3]是train/test/valid
    # row[4]是学名（可以不用管）
    name = './birds.csv'
    data = pd.read_csv(name)
    labels = data['labels'].tolist()

    label_mapping = {}
    label_list = []
    numeric_labels = []
    count = 0

    for label in labels:
        if label not in label_mapping:
            label_mapping[label] = count
            label_list.append(label)
            count += 1
        numeric_labels.append(label_mapping[label])
        if count == class_num:
            break
    print("step1: 获得labels_dic成功！")
    print(label_list)

    # step 2: 读取数据
    train_data = data[data['data set'] == 'train']
    train_data = train_data[train_data['class id'] < class_num]
    val_data = data[data['data set'] == 'valid']
    val_data = val_data[val_data['class id'] < class_num]
    test_data = data[data['data set'] == 'test']
    test_data = test_data[test_data['class id'] < class_num]

    X_train = train_data['filepaths'].tolist()  
    y_train_string = train_data['labels'].tolist()
    y_train = [label_mapping[label] for label in y_train_string]

    X_val = val_data['filepaths'].tolist()  
    y_val_string = val_data['labels'].tolist()  
    y_val = [label_mapping[label] for label in y_val_string]

    X_test = test_data['filepaths'].tolist()  
    y_test_string = test_data['labels'].tolist()  
    y_test = [label_mapping[label] for label in y_test_string]

    train_dataset = ImageDataset(X_train, y_train, transform)
    val_dataset = ImageDataset(X_val, y_val, transform)
    test_dataset = ImageDataset(X_test, y_test, transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False)
    
    print("step2: 数据读取、转换成功！")
    print(val_data)
    return label_mapping, train_loader, val_loader, test_loader


