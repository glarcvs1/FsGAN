import glob,json
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torchvision.models.resnet import resnet18
from torchvision.models.mobilenet import MobileNetV2

# 定义模型
class res18(nn.Module):
    def __init__(self,n=1000):
        super(res18, self).__init__()
        self.cnn = models.resnet18(pretrained=False)  # 加载resnet50
        self.cnn.avgpool = nn.AdaptiveAvgPool2d(1)  # 将平均池化改为自适应平均池化
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 去除最后的线性层
        #self.cla=cla
        self.fc1= nn.Linear(512,n)
        #self.fc2 = nn.Linear(50, 2)
    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1= self.fc1(feat)
        #c2=self.fc2(c1)
        return c1

class res50(nn.Module):
    def __init__(self,n=1000):
        super(res50, self).__init__()
        self.cnn = models.resnet50(pretrained=True)  # 加载resnet50
        self.cnn.avgpool = nn.AdaptiveAvgPool2d(1)  # 将平均池化改为自适应平均池化
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 去除最后的线性层
        #self.cla=cla
        self.fc1= nn.Linear(2048, n)
        #self.fc2 = nn.Linear(50, 2)
    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1= self.fc1(feat)
        #c2=self.fc2(c1)
        return c1

class res50(nn.Module):
    def __init__(self,n=1000):
        super(res50, self).__init__()
        self.cnn = models.resnet50(pretrained=False)  # 加载resnet50
        self.cnn.avgpool = nn.AdaptiveAvgPool2d(1)  # 将平均池化改为自适应平均池化
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 去除最后的线性层
        #self.cla=cla
        self.fc1= nn.Linear(2048, n)
        #self.fc2 = nn.Linear(50, 2)
    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1= self.fc1(feat)
        #c2=self.fc2(c1)
        return c1

class res_test(nn.Module):
    def __init__(self):
        super(res_test, self).__init__()
        self.cnn = models.resnet101(pretrained=False)  # 加载resnet50
        # self.cnn.avgpool = nn.AdaptiveAvgPool2d(1)  # 将平均池化改为自适应平均池化
        #self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 去除最后的线性层
        # #self.cla=cla
        # self.fc1= nn.Linear(2048, 2)
        # #self.fc2 = nn.Linear(50, 2)
    def forward(self, img):
        feat = self.cnn(img)
        # feat = feat.view(feat.shape[0], -1)
        # c1= self.fc1(feat)
        # #c2=self.fc2(c1)
        return feat
if __name__ == "__main__":
    net=res18(6)
    print(net)
    #net1 = res_test()
    #print(net1)
    net.fc1.out_features=2
    print(net)