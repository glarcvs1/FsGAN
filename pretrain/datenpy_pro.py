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
            self.data = data[leib:leie, :tb, :, :, :]
            self.prodata = prodata[:,:l,:,:,:]
            self.k = 1
            self.imgs= self.make_traindata(self.data,self.prodata)
        else:
            self.data = data[leib:leie, :tb, :, :, :]
            self.imgs = self.make_valdata(self.data)
        if mode ==1:# 验证
            self.data = data[leib:leie, tb:tn, :, :, :]
            self.imgs=self.make_valdata(self.data)
        if mode == 2:  # 测试
            self.data = data[leib:leie, tn:, :, :, :]
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
        return self.leie-self.leib
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
    dir="input\\animal_128.npy"
    dir1 = "input\\vggface_pro2000_6.npy"
    dir2 = "vgg_face_data_rgb.npy"
    prove_dir="input\\vgg_face_data.npy"
    data1 = np.load(dir2)
    #data = np.hstack((data[1:3, :1, :, :, :], data[1:3 ,3:, :, :, :]))
    print(data1.shape)
    i=data1[2][22]
    print(i)
    #data=getdata(dir,1,2,20,30,65,prove_dir,200)
    #datatest = getdata(dir, 2, 10, 20, 1, 10)
    #print(data.getlen())
    #print(datatest.getlen())
    #i, j = data[299]
    plt.imshow(i)
    plt.show()