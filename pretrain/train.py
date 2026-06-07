import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import argparse
import copy
import resnet
from tqdm import tqdm
import pandas as pd
import datenpy as npy
#import ninsdata as nins
#import dcemridata_withnpy as dce
#import roc
from scipy.stats import scoreatpercentile
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import  torch.nn.functional as F
# 训练模型的函数
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #target=[]
    #proba=[]
    traintransforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valtransforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    traindata = npy.getdata(args.dir, args.mode, args.tb,args.te,args.leib,args.leie,args.prodir, transform=traintransforms)
    valdata = npy.getdata(args.dir, args.mode+1, args.tb,args.te,args.leib,args.leie, transform=valtransforms)
    traindataloader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valdataloader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=True, num_workers=0)
    trainlen = traindata.getlen()
    vallen = valdata.getlen()
    lei = traindata.getlei()
    model = resnet.res18(lei)
    #是否存在初始参数
    if args.weights != "" :
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                    if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))
    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # 对所有网络层参数进行更新
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 学习率策略、，每 7 个 epochs 乘以 0.1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.s_size, gamma=args.gam)
    since = time.time()
    keep_acc = 0.0
    keep_loss = 10
    #ter_cor=10
    #e = 20
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1),end='   ')
        #print('-' * 10)
        # 每个 epoch 都分为训练阶段和验证阶段
        for phase in ['train', 'val']:
            # 注意训练和验证阶段，需要分别对 model 的设置
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            if phase == 'train':
                dataloader=traindataloader
                lens=trainlen
            else:
                dataloader=valdataloader
                lens=vallen
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 清空参数的梯度
                optimizer.zero_grad()
                # 只有训练阶段才追踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(outputs)
                    loss = criterion(outputs, labels)
                    #outputs = F.softmax(outputs)
                    #print(outputs)
                    pred, preds = torch.max(outputs, 1)
                    # 训练阶段才进行反向传播和参数的更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # 记录 loss 和 准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #if(phase=='val'): #保存模型
                    #print('save')
            epoch_loss = running_loss / lens
            epoch_acc = running_corrects.double() / lens
            if (phase == 'train'):
                train_loss = epoch_loss
            if (phase == 'val'):
                val_loss = epoch_loss
                val_acc=epoch_acc
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc),end='   ')
            # deep copy the model
            if phase == 'val' and val_acc > keep_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                keep_acc=val_acc

        time_elapsed =time.time() - since
        print('时间:{:.1f} s'.format(time_elapsed))

        strs='./output/'+args.msave+str(num_epochs)+'.pkl'
        torch.save(best_model_wts, strs)

if __name__ == "__main__":
    #dced=['AATH','DP','BRIX','EXTOFTS','TOFTS']
    num_epochs=100
    #labol = 'res50_pro239_' + str(num_epochs)+'/' #roc曲线文件命名标志
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_size', type=int, default=10,help='多少步更新')
    #parser.add_argument('--num_classes', type=int, default=5) #类别数
    parser.add_argument('--mode', type=int, default=0) #训练集还是验证
    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--gam', type=float, default=0.1)
   # parser.add_argument('--dced', type=list, default=['AATH','DP','BRIX','ETOFTS','TOFTS'])
    parser.add_argument('--batch_size', type=int, default=16, help='批次的大小')
    parser.add_argument('--p', type=float, default=0.01, help='类别的比例')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--dex', type=int, default=0)
    parser.add_argument('--tb', type=int, default=75,help='ani=30,fllow=12,dce=75') #验证集开始
    parser.add_argument('--te', type=int, default=125,help='ani=65,fllow=26,dce=125')  # 验证集结束，测试集开始
    parser.add_argument('--leib', type=int, default=16,help='ani=119,fllow=85,dce=16,18,20,22,24')  # 类别开始
    parser.add_argument('--leie', type=int, default=18,help='ani=150,fllow=103,dce=18,20,22,24,26')  # 类别结束
    parser.add_argument('--dir', type=str,
                        default="input/dce125.npy",help='数据集的地址')
    parser.add_argument('--msave', type=str,
                        default="dce_res18_", help='模型保存名称')
    parser.add_argument('--prodir', type=str,
                        default="input/flower_pro6.npy", help='数据集的地址')
    parser.add_argument('--weights', type=str, default='',
                        help='模型的初始参数')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='是否要冻结某些层')
    opt = parser.parse_args()
    main(opt)