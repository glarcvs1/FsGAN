import cv2
from pylab import *  # 支持中文

mpl.rcParams['font.sans-serif'] = ['SimHei']

img2 = cv2.imread("img/fenge-1-1.jpg")
img4 = cv2.imread("img/fenge-1-1-0.06.jpg")
img5 = cv2.imread("img/fenge-1-1-0.02.jpg")
img6 = cv2.imread("img/fenge-1-1-0.1.jpg")

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
h1, s, img2 = cv2.split(img2)  # 看单通道的
img_array2 = np.array(img2)  # 把图像转成数组格式img = np.asarray(image)
mean2 = np.average(img_array2, axis=0)  # 按列求均值

img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)
h1, s, img4 = cv2.split(img4)  # 看单通道的
img_array4 = np.array(img4)  # 把图像转成数组格式img = np.asarray(image)
mean4 = np.average(img_array4, axis=0)  # 按列求均值

img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2HSV)
h1, s, img5 = cv2.split(img5)  # 看单通道的
img_array5 = np.array(img5)  # 把图像转成数组格式img = np.asarray(image)
mean5 = np.average(img_array5, axis=0)  # 按列求均值

img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2HSV)
h1, s, img6 = cv2.split(img6)  # 看单通道的
img_array6 = np.array(img6)  # 把图像转成数组格式img = np.asarray(image)
mean6 = np.average(img_array6, axis=0)  # 按列求均值

# plt.plot(mean2,c="g",label=u'a=std/(255*2)')
# plt.plot(mean4,linestyle=":",c="r",label=u'a=0.06')
# plt.plot(mean5,linestyle="-.",c="b",label=u'a=0.02')
# plt.plot(mean6,linestyle="--",c="k",label=u'a=0.1')

plt.plot(mean2, linewidth=1, c="g", label=u'a=std/(255*2)')
plt.plot(mean4, linewidth=1, linestyle=":", c="r", label=u'a=0.06')
plt.plot(mean5, linewidth=1, linestyle="--", c="b", label=u'a=0.02')
plt.plot(mean6, linewidth=1, linestyle="-.", c="darkorange", label=u'a=0.1')

plt.legend(loc='upper right', prop={'family': 'SimHei', 'size': 9})  # 让图例生效
# plt.legend(loc='upper right',size=5)  # 让图例生效lower
plt.tick_params(direction='in')  # 刻度向里
plt.xlim([0, 255])  # 横坐标范围
plt.ylim([100, 150])  # 横坐标范围
plt.ylim([100, 150])  # 横坐标范围
# plt.grid(True)  ##增加格点
plt.axis('tight')  # 坐标轴适应数据量 axis 设置坐标轴
plt.xlabel("图像1宽度/（像素）", size=10)
plt.ylabel("列平均灰度值", size=10)

# plt.title("a不同取值的L*(x,y)的一维曲线图")  # 标题

# 方法一
# plt.rcParams['savefig.dpi'] = 800 #图片像素
# #
# # plt.rcParams['figure.dpi'] = 800 #分辨率
# #
# # # plt.axis('off')可以去坐标轴
# # plt.savefig('img/name-2.jpg')
# 方法二
# 可以直接设置保存好的图的清晰度，大小
# plt.set_size_inches(6, 6)  #设置保存图片的尺寸
plt.savefig('img/name-1.jpg', dpi=800, bbox_inches='tight')

# bbox_inches使生成的图片周围的空白缩小
# 在 plt.show() 之前调用 plt.savefig()，否则出现空白

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()