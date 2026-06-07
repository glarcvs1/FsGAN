#from   import autograd, npx,np
import matplotlib.pyplot as plt
#import npx
import numpy as np


def relu(x):
    return np.maximum(0,x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return ((1 -np.exp(-2*x))/ (1 + np.exp(-2*x)))





















if __name__ == "__main__":
    #npx.set_np()
    x = [0,50,100,200,400,800,1200,2000]
    #x.attach_grad()
    #with autograd.record():
    y = [0.5844,0.6409,0.6665,0.6758,0.7101,0.7327,0.7540,0.7653]
    x1=[0,50,100,200,400,800,1200,2000]
    y1=[0.5906,0.6263,0.6418,0.6613,0.6793,0.6863,0.6869,0.6941]
    #plt.set_size_inches(6, 6)
    plt.ylim(0.40, 0.90,0.1)
    #plt.ylim(0, 1200, 50)
    plt.figure(figsize=(12, 6))
    x_ticks = range(0, 2100, 200)
    plt.xticks(ticks=x_ticks)
    plt.xlabel('num',fontsize=14)
    plt.ylabel('acc',fontsize=14)
    #plt.figure(figsize=(10, 6))
    plt.rcParams['savefig.dpi'] = 1200 # 图片像素
    #plt.rcParams['figure.dpi'] = 2000  # 分辨率
    #plt.plot(x, y)
    plt.plot(x, y, color='darkred',
             lw=2, label='ours')
    plt.plot(x1, y1, color='tan',
             lw=2, label='LoFGAN')
    plt.title("Flower",fontsize=14)
    plt.legend()
    #plt.title("a不同取值的L*(x,y)的一维曲线图")  # 标题
    plt.savefig('flower—acc''.jpg', bbox_inches='tight')
    plt.show()
   # plt.title("a不同取值的L*(x,y)的一维曲线图")  # 标题
    #plt.save()