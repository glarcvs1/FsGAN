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

if __name__ == "__main__":
    dir="input\\animal_128.npy"
    prove_dir="input\\vgg_face_data.npy"
    data1 = np.load(prove_dir)
    #data = np.hstack((data[1:3, :1, :, :, :], data[1:3 ,3:, :, :, :]))
    print(data1.shape)
    i=data1[2][22]
    data2 = data1[:, :,:, :, ::-1]
    data2 = (data2 * 255).astype(np.uint8)
    np.save("vgg_face_data_rgb.npy", data2)
    i = data2[2][22]
    #data=getdata(dir,1,2,20,30,65,prove_dir,200)
    #datatest = getdata(dir, 2, 10, 20, 1, 10)
    #print(data.getlen())
    #print(datatest.getlen())
    #i, j = data[299]
    plt.imshow(i)
    plt.show()