#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# 指定根目录，可以根据需要修改为实际路径，例如："/home/user/dataset"
base_dir = "/dataset/zhuluoji/unet/unet_double"  

# 定义需要创建的目录路径
jpeg_images_dir = os.path.join(base_dir, "VOCdevkit_dilate", "VOC2007", "JPEGImages")
segmentation_class_dir = os.path.join(base_dir, "VOCdevkit_dilate", "VOC2007", "SegmentationClass")

# 创建目录结构
os.makedirs(jpeg_images_dir, exist_ok=True)
os.makedirs(segmentation_class_dir, exist_ok=True)

print("目录结构创建完成！")