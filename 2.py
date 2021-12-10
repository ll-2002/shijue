#���뺯����
import cv2
import numpy as np
#��ȡһ��Ӳ��ͼ��
img=cv2.imread("coins2.jpg")
#��ͨ�˲�����
#��ͼ����з��鴦��
h, w = img.shape[:2] #��ȡͼ��ĳ��Ϳ�
mask = np.zeros((h+2, w+2), np.uint8)#����ͼ�����
cv2.floodFill(img, mask, (w-1,h-1), (255,255,255), (2,2,2),(3,3,3),8) 
#ͼ��ҶȻ�
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ͨ����˹�˲���ͼ�����ģ�������������Ϊ��ͼ��Ӳ��ȥ��
blur=cv2.cv2.GaussianBlur(gray,(29,29),0,0)#�����������ֵ�˲��������Ӷ�ͼ��Ч��ѡ��
cv2.imshow("blur",blur)
cv2.waitKey(0)
#ͨ����������ֵ����ͼ�������ֵ��������Ӳ����������Χ�������ֿ���
ret,thresh1=cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
cv2.imshow("thresh1",thresh1)
cv2.waitKey(0)
#���б����㣬ȥ��ͼ���ڲ�����
kernel = np.ones((7,7), np.uint8)#���þ����
close=cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)#������
cv2.imshow("close",close)
cv2.waitKey(0)
#����canny�㷨��ͼ�����������ȡ
Canny = cv2.Canny(close, 20, 150)
#��ʾͼ��������ȡͼ��
cv2.imshow("Canny",Canny)
#�ȴ����̼�ֵ�ر�
cv2.waitKey(0)
#����ȡ��������ͼ�����ҳ���������
(cnts1,_) =cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#���ն˴�ӡ��Ӳ�Ҹ���
print("ͼ���е�Ӳ�ҹ���:",len(cnts1),"��")
#��Ӳ��������������ԭͼ
coins1 = img.copy()
cv2.drawContours(coins1, cnts1, -1, (0,255,0), 2)
#��ʾ��ԭͼ�����������Ӳ��
cv2.imshow("coins",coins1)
#�ȴ����̼�ֵ�ر�
cv2.waitKey(0)
