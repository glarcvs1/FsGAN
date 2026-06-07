
import matplotlib.pyplot as plt
#import npx
import numpy as np



if __name__ == "__main__":
    #npx.set_np()
    x = [5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
    #x.attach_grad()
    #with autograd.record():
    y = [0.051038574, 0.048593946, 0.04473712, 0.051296078, 0.063476905, 0.052564802, 0.0528646, 0.051991414, 0.050244965, 0.050035812, 0.052964102]
    x5 = [5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
    y5=[0.033259798, 0.04049326,  0.03550041, 0.036379877, 0.040849764, 0.04379208, 0.04252636,0.041857943, 0.04298634,  0.04208088,  0.04332298]
    x7 = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    y7 = [0.025892204,  0.03388924, 0.033880968,  0.031497777, 0.044243257,  0.045878462, 0.046459552, 0.04548621, 0.045084488, 0.04579495,  0.047041703]
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
    plt.title("LIPIS",fontsize=14)
    plt.legend()
    #plt.title("a不同取值的L*(x,y)的一维曲线图")  # 标题
    plt.savefig('LIPIS''.jpg', bbox_inches='tight')
    plt.show()
   # plt.title("a不同取值的L*(x,y)的一维曲线图")  # 标题
    #plt.save()