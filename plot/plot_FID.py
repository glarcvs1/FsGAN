
import matplotlib.pyplot as plt
#import npx
import numpy as np



if __name__ == "__main__":
    #npx.set_np()
    x = [5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
    #x.attach_grad()
    #with autograd.record():
    y = [89.35699064307354,  55.846901172325516, 41.11695888650817,26.905259369698143,29.2491512478282,30.501647484179244,23.626746359458394,21.69866572443243,24.993488396854616,22.29168507643638,20.32212434432185]
    x5 = [5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
    y5=[176.51900972788997, 62.50619003928625, 35.800845661449785,  32.627964923318785, 34.95812193317121, 37.093068919236345,28.45245004978682, 34.9027361045346,   28.342036835284574, 27.232897893550643, 26.063269240013028]
    x7 = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    y7 = [133.1918216816239, 90.01673199141885,  39.67058788638866,  37.52878868696746, 28.99477395011408, 34.71153750044559,   30.135675324201856,    27.676707450205726,  28.895595447866384,  24.551516289301105,  28.704265126055162]
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
    plt.title("FID",fontsize=14)
    plt.legend()
    #plt.title("a不同取值的L*(x,y)的一维曲线图")  # 标题
    plt.savefig('FID''.jpg', bbox_inches='tight')
    plt.show()
   # plt.title("a不同取值的L*(x,y)的一维曲线图")  # 标题
    #plt.save()