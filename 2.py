#导入函数库
import cv2
import numpy as np
#读取一张硬币图像
img=cv2.imread("coins2.jpg")
#低通滤波处理
#对图像进行泛洪处理
h, w = img.shape[:2] #获取图像的长和宽
mask = np.zeros((h+2, w+2), np.uint8)#进行图像填充
cv2.floodFill(img, mask, (w-1,h-1), (255,255,255), (2,2,2),(3,3,3),8) 
#图像灰度化
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#通过高斯滤波对图像进行模糊处理，可以理解为对图像硬币去噪
blur=cv2.cv2.GaussianBlur(gray,(29,29),0,0)#这里可以用中值滤波，具体视对图像效果选择
cv2.imshow("blur",blur)
cv2.waitKey(0)
#通过二进制阈值化对图像进行阈值化处理，将硬币轮廓与周围噪声区分开来
ret,thresh1=cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
cv2.imshow("thresh1",thresh1)
cv2.waitKey(0)
#进行闭运算，去除图像内部噪声
kernel = np.ones((7,7), np.uint8)#设置卷积核
close=cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)#闭运算
cv2.imshow("close",close)
cv2.waitKey(0)
#利用canny算法对图像进行轮廓提取
Canny = cv2.Canny(close, 20, 150)
#显示图像轮廓提取图像
cv2.imshow("Canny",Canny)
#等待键盘键值关闭
cv2.waitKey(0)
#在提取出的轮廓图像中找出轮廓线条
(cnts1,_) =cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#在终端打印出硬币个数
print("图像中的硬币共有:",len(cnts1),"个")
#将硬币轮廓线条画在原图
coins1 = img.copy()
cv2.drawContours(coins1, cnts1, -1, (0,255,0), 2)
#显示在原图上面检测出来的硬币
cv2.imshow("coins",coins1)
#等待键盘键值关闭
cv2.waitKey(0)
