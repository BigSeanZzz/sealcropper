import cv2
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import os
import sys
import time
import random

def crop_image(image_path, filename, destination_path):
    # log_output = 1：打印过程，调试时用。
    # log_output = 0：不打印过程，批量处理时用。
    log_output = 0

    #imread 返回的是BGR格式的图像，与 RGB 仅字节顺序相反
    if os.path.isfile(image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    else:
        print("The file " + image_path + " does not exist.")

    if log_output == 1:
        plt.figure('opencv imgs')
        plt.subplot(431)
        plt.imshow(image)
        print('image shape: ', image.shape)

    # 白色以外颜色
    # filter = [([0, 0, 0], [230, 230, 230])]

    # 白色和黄色以外颜色
    filter = [([0, 0, 0], [200, 200, 200])]

    # 去除阴影
    filter = [([0, 0, 0], [150, 150, 150])]


    # 如果color中定义了几种颜色区间，都可以分割出来
    for (lower, upper) in filter:
        # 创建NumPy数组，传入参数即为数组元素
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限

        # 根据阈值找到对应颜色
        mask = cv2.inRange(image, lower, upper) #二值化的图
        #自身按位相与，取mask部分
        image_red = cv2.bitwise_and(image, image, mask=mask)

        # 展示图片
        if log_output == 1:
            plt.subplot(432)
            plt.imshow(image_red, cmap='gray')

    image_gray = cv2.cvtColor(image_red, cv2.COLOR_BGR2GRAY)
    if log_output == 1:
        plt.subplot(433)
        plt.imshow(image_gray, cmap='gray')

    # 用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。
    gradX = cv2.Sobel(image_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(image_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    if log_output == 1:
        plt.subplot(434)
        plt.imshow(gradient, cmap='gray')
    image_gradient = cv2.convertScaleAbs(gradient)
    if log_output == 1:
        plt.subplot(435)
        plt.imshow(image_gradient, cmap='gray')

    # 首先使用低通滤泼器平滑图像（9 x 9内核）
    # blur and threshold the image
    image_blurred = cv2.blur(image_gradient, (15, 15))
    (_, image_thresh) = cv2.threshold(image_blurred, 60, 255, cv2.THRESH_BINARY)
    if log_output == 1:
        plt.subplot(436)
        plt.imshow(image_blurred, cmap='gray')
        plt.subplot(437)
        plt.imshow(image_thresh, cmap='gray')

    # 白色填充这些空余，使得后面的程序更容易识别区域，这需要做一些形态学方面的操作。
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    image_closed = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, kernel)
    if log_output == 1:
        plt.subplot(438)
        plt.imshow(image_closed, cmap='gray')

    # 白色斑点，这会干扰之后的轮廓的检测，要把它们去掉。分别执行4次形态学腐蚀与膨胀。
    # perform a series of erosions and dilations
    image_closed = cv2.erode(image_closed, None, iterations=4)
    image_closed = cv2.dilate(image_closed, None, iterations=4)
    if log_output == 1:
        plt.subplot(439)
        plt.imshow(image_closed, cmap='gray')
        print('closed shape: ', image_closed.shape)

    #图像二值化
    ret, image_closed = cv2.threshold(image_closed ,127, 255 ,cv2.THRESH_BINARY)
    if log_output == 1:
        plt.subplot(4,3,10)
        plt.imshow(image_closed, cmap='gray')

    #轮廓提取
    # 老版本函数
    # image_closed, contours, hierarchy = cv2.findContours(image_closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(image_closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plt.show()
    #进行图像的截取和保存
    plt.figure('cropImage')
    counter = 0
    for c in contours:
        #排除最外面的图像框
        # if c.shape[0] < 5:
        #     print('shape has too less points:', c.shape)
        #     continue

        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        print('box:', box)
        cv2.drawContours(image_closed, [box], -1, (200, 200, 200), 3)

        # 图像截取
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1

        #滤除图像大太小的图片
        if hight < 1000 or width < 1000:
            continue
        #将截取范围稍微缩小
        # cropImage = image[int(y1-hight*0.1):int(y1+hight*1.1), int(x1-width*0.1):int(x1+width*1.1)]
        cropImage = image[int(y1+0.2*hight):int(y2-0.2*hight), int(x1+width*0.2):int(x2-width*0.2)]
        counter += 1
        if log_output == 1:
            plt.subplot(3, 3, counter)
            plt.imshow(cropImage)
        #保存图片
        cropImage_filename = filename.split('.')[-2] + '_' + str(counter) + '.jpg'
        cropImage_filepath = os.path.join(destination_path, cropImage_filename)
        print('filename:', filename.split('.')[-2])
        print('cropImage_filename:', cropImage_filename)
        print('cropImage_filepath:', cropImage_filepath)
        cv2.imwrite(cropImage_filepath, cropImage)

    if log_output == 1:
        plt.figure('opencv imgs')
        plt.imshow(image_closed)
        plt.show()

def main(argv=None):
    destination_path = './des'
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    #待处理文件根目录
    dir_path = './raw'
    #打开根目录
    file_dir = os.listdir(dir_path)
    print('file_dir:', file_dir)

    fileflow = 0
    for filename_image in file_dir:
        if filename_image.split('.')[-1] != 'jpg' and filename_image.split('.')[-1] != 'JPG':
            continue
        print('filename_image:', filename_image)
        fileflow_name = 'img' + str(fileflow) + '-'+ str(random.randint(1000000, 9999999)) + '.jpg'
        print('fileflow_name:', fileflow_name)
        image_path = os.path.join(dir_path, filename_image)
        # image_path = uncode(image_path, "utf8")
        print('image_path:', image_path)
        crop_image(image_path, fileflow_name, destination_path)
        fileflow += 1

if __name__ == '__main__':
    main()

