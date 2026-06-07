
import matplotlib.pyplot as plt
#import npx
import numpy as np



if __name__ == "__main__":
    #npx.set_np()
    x = [5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
    #x.attach_grad()
    #with autograd.record():
    y = [1.9756842408698794, 1.7501852596884142, 1.6488554132365585, 1.735770701967294, 1.8110840207727699, 1.6594046715864124, 1.6970634392182227, 1.6941994709225945, 1.6691139507825, 1.6848024673236548, 1.7242995414852031]
    x5 = [5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
    y5=[1.8336764168485913,1.4629440237530305,  1.6257982916571474, 1.5751963392288053,  1.737658534241588, 1.6291533032429009, 1.729452513234931, 1.6542656811624508, 1.779972071686452, 1.6812674113106887,  1.695190880300387]
    x7 = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    y7 = [1.6893780520954913,  1.6896263649919903, 1.5716433709763877, 1.480847036923412,1.5391251847974547,  1.504551917833841, 1.528796199801172, 1.615957753283152, 1.6367041645360936,  1.668746205896813, 1.6625534268802302]
    #plt.set_size_inches(6, 6)
    plt.ylim([0.025, 0.90])
   # plt.ylim(0, 1200, 50)
    plt.figure(figsize=(12, 6))
    x_ticks = range(5000, 100000, 10000)
    plt.xticks(ticks=x_ticks)
    plt.xlabel('iter',fontsize=14)
    #plt.ylabel('acc',fontsize=14)
    plt.xlim(5000, 100000)
    #fig.updata_xaxes(range=(5000,100000))
    #plt.figure(figsize=(10, 6))
    plt.rcParams['savefig.dpi'] = 300 # 图片像素
    #plt.rcParams['figure.dpi'] = 2000  # 分辨率
    #plt.plot(x, y)
    plt.plot(x, y, color='darkred',
             lw=2, label='3-shot')
    plt.plot(x5, y5, color='darkgreen',
             lw=2, label='5-shot')
    plt.plot(x7, y7, color='darkblue',
             lw=2, label='7-shot')
    plt.title("IS",fontsize=14)
    plt.legend()
    #plt.title("a不同取值的L*(x,y)的一维曲线图")  # 标题
    plt.savefig('IS''.jpg', bbox_inches='tight')
    plt.show()
   # plt.title("a不同取值的L*(x,y)的一维曲线图")  # 标题
    #plt.save()