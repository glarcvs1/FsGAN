import os
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets.folder import is_image_file, accimage_loader, pil_loader
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
#import accimage
import csv
import random

class getdata(Dataset):
    def __init__(self,root_dir,mode,tb,tn,leib,leie,prove_dir=None,l=None,transform=None):
        #tb表示验证数据开始位置
        #tn表示测试数据开始位置
        super(getdata, self).__init__()
        self.transform = transform
        self.leib=leib
        self.leie = leie
        data=np.load(root_dir)
        if prove_dir!=None:
            prodata=np.load(prove_dir)
        self.tb=tb
        a5,b5,c5,d5,e5=data.shape
        self.datalen=b5
        if mode==0 and prove_dir!=None:   #训练
            self.data = np.concatenate((data[:leib, :tb, :, :, :],data[leie:, :tb, :, :, :]),axis=0)
            self.prodata = np.concatenate((prodata[:leib, :l, :, :, :],prodata[leie:, :l, :, :, :]),axis=0)
            self.k = 1
            self.imgs= self.make_traindata(self.data,self.prodata)
        else:
            self.data = np.concatenate((data[:leib, :tb, :, :, :],data[leie:, :tb, :, :, :]),axis=0)
            self.imgs = self.make_valdata(self.data)
        if mode ==1:# 验证
            self.data = np.concatenate((data[:leib, tb:tn, :, :, :],data[leie:, tb:tn, :, :, :]),axis=0)
            self.imgs=self.make_valdata(self.data)
        if mode == 2:  # 测试
            self.data = np.vstack(data[:leib, tn:, :, :, :],data[leie:, tn:, :, :, :])
            self.imgs = self.make_valdata(self.data)
    def make_traindata(self,data,prodata):
        images = []
        a,b, c, d, e = data.shape
        a2, b2, c2, d2, e2 = prodata.shape
        for i in range(a2):
            for j in range(b2):
                imgs=prodata[i,j,:,:,:]
                item = (imgs, i)
                images.append(item)
        for i in range(a):
            for j in range(b):
                imgs = data[i, j, :, :, :]
                item = (imgs, i)
                images.append(item)
        return images

    def make_valdata(self,data):
        images=[]
        a,b,c,e,d=data.shape
        for i in range(a):
            for j in range(b):
                imgs = data[i, j, :, :, :]
                item = (imgs, i)
                images.append(item)
        return images

    def getlei(self):
        return 24
    def __getitem__(self, item):
        img, cls = self.imgs[item]
        img = Image.fromarray(img)
        if self.transform is not None:
            img=self.transform(img)
        return img, cls

    def getlen(self):
        return len(self.imgs)

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    dir="input\\dce3.npy"
    dir2 = "input\\dce125.npy"
    prove_dir="input\\dce_pro2000_6.npy"
    data1 = np.load(prove_dir)
    #data = np.hstack((data[1:3, :1, :, :, :], data[1:3 ,3:, :, :, :]))
    print(data1.shape)
    data=getdata(dir,0,10,120,0,20)
    data2 = getdata(dir2, 0, 10, 120, 0, 4)
    #datatest = getdata(dir, 2, 10, 20, 1, 10)
    print(data2.getlen())
    #print(datatest.getlen())
    i, j = data2[70]
    plt.imshow(i)
    plt.show()