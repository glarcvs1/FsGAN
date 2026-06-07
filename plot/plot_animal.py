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
    y = [0.2064,0.2901,0.3126,0.3405,0.3816,0.4154,0.4240,0.4334]
    x1=[0,50,100,200,400,800,1200,2000]
    y1=[0.2052,0.2636,0.2709,0.2962,0.3262,0.3296,0.3334,0.3531]
    #plt.set_size_inches(6, 6)
    plt.ylim(0.2, 0.5,0.1)
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
    plt.title("Animal Face",fontsize=14)
    plt.legend()
    #plt.title("a不同取值的L*(x,y)的一维曲线图")  # 标题
    plt.savefig('animal—acc''.jpg', bbox_inches='tight')
    plt.show()
   # plt.title("a不同取值的L*(x,y)的一维曲线图")  # 标题
    #plt.save()
