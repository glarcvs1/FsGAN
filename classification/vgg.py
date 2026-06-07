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
class vgg_16(nn.Module):
    def __init__(self,lei):
        super(vgg_16, self).__init__()
        net = models.vgg16(pretrained=False)
        # print(net)
        # 在容器中添加多个网络层,以顺序的方式包装一组网络
        #net.classifier = nn.Sequential()
        #self.features = net
        net.classifier[6] = nn.Linear(in_features=4096, out_features=lei)
        self.features = net
        #self.features = nn.Sequential(*list(net.children())[:-1])
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(0.8),
        #     #nn.Linear(4096, 4096),
        #     #nn.ReLU(True),
        #     #nn.Dropout(0.8),
        #     nn.Linear(4096, lei)
        # )

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x

class vgg_test(nn.Module):
    def __init__(self):
        super(vgg_test, self).__init__()
        net = models.vgg16(pretrained=True)
        # print(net)
        # 在容器中添加多个网络层,以顺序的方式包装一组网络
        #net.classifier = nn.Sequential()
        self.features = net
        #self.features = nn.Sequential(*list(net.children())[:-1])
        net.classifier[6] = nn.Linear(in_features=4096, out_features=10)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(0.8),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(0.8),
        #     nn.Linear(4096, 2)
        # )

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x

if __name__ == "__main__":
    net=vgg_16(10)
    print(net)
    net1 = vgg_test()
    print(net1)