import cv2
import pyzbar.pyzbar as pyzbar
import numpy
from PIL import Image, ImageDraw, ImageFont


def decodeDisplay(img_path):

    img_data = cv2.imread(img_path)
    # תΪ�Ҷ�ͼ��
    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    barcodes = pyzbar.decode(gray)

    for barcode in barcodes:

        # ��ȡ������ı߽���λ��
        # ����ͼ����������ı߽��
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # ����������Ϊ�ֽڶ���������������������ͼ����
        # ������������Ҫ�Ƚ���ת�����ַ���
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        #������ʾ����
        # ���ͼ��������������ݺ�����������
        #text = "{} ({})".format(barcodeData, barcodeType)
        #cv2.putText(imagex1, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,5, (0, 0, 125), 2)

        #����Ϊ��
        img_PIL = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
        # ���������壬Ĭ�ϴ�С��
        font = ImageFont.truetype('msyh.ttc', 35)
        # ������ɫ��rgb)
        fillColor = (0, 255, 255)
        # �������λ��
        position = (x, y-50)
        # �������
        str = barcodeData
        # ��Ҫ�Ȱ�����������ַ�ת����Unicode������ʽ(  str.decode("utf-8)   )

        draw = ImageDraw.Draw(img_PIL)
        draw.text(position, str, font=font, fill=fillColor)
        # ʹ��PIL�е�save��������ͼƬ������
        img_PIL.save('1.jpg', 'jpeg')
        # ���ն˴�ӡ���������ݺ�����������
        print("ɨ����==�� ��� {0} ���ݣ� {1}".format(barcodeType, barcodeData))

if __name__ == '__main__':
    decodeDisplay("ma.jpg")
