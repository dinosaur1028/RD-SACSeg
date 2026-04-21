#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os

# 设定原始掩码文件夹和保存结果的文件夹
input_folder = "/dataset/zhuluoji/unet/unet_double/VOCdevkit3_copy/VOC2007/SegmentationClass"       # 存放原始掩码图的文件夹
output_folder = "/dataset/zhuluoji/unet/unet_double/VOCdevkit3_copy/VOC2007/SegmentationClass_dilated_3x3_255"  # 保存膨胀后图像的文件夹
output_folder2 = "/dataset/zhuluoji/unet/unet_double/VOCdevkit3_copy/VOC2007/SegmentationClass_dilated_3x3"# 保存膨胀后掩码为1
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(output_folder2):
    os.makedirs(output_folder2)

# 构造结构元素（使用椭圆形核，大小为7x7，对应半径3）
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# 遍历文件夹中所有图像
for filename in os.listdir(input_folder):
    # 这里假设图像后缀为 .png，可根据实际情况修改
    if filename.lower().endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        # 读取图像为灰度图
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("无法读取文件:", img_path)
            continue

        # 如果图像的像素值不是严格的0和1，可以先做二值化处理：
        # 这里假设大于127视为1，否则为0
        _, bin_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)

        # 进行膨胀操作
        dilated = cv2.dilate(bin_img, kernel, iterations=1)
        # 注意：膨胀后的结果依然是0和1，如需保存为常规图像（像素值0-255）则可以乘以255
        dilated_uint8 = (dilated * 255).astype("uint8")

        # 保存结果图像
        output_path = os.path.join(output_folder, filename)
        output_path2 = os.path.join(output_folder2, filename)

        cv2.imwrite(output_path, dilated_uint8)
        cv2.imwrite(output_path2, dilated)
        print(f"处理并保存图像: {output_path}")