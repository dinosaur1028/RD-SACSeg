#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil

def extract_images_and_masks(voc_dir, output_dir):
    """
    voc_dir: VOC 数据集根目录，即 /dataset/zhuluoji/unet/unet_double/VOCdevkit3/VOC2007
    output_dir: 提取后的数据存放目录
    """
    # 定义原图和掩码图所在目录
    images_src = os.path.join(voc_dir, 'JPEGImages_Aug')
    masks_src  = os.path.join(voc_dir, 'SegmentationClass_Aug')
    
    # ImageSets/Segmentation 下存放 txt 文件
    imgset_dir = os.path.join(voc_dir, 'ImageSets', 'Segmentation')
    subset_files = ['train.txt', 'val.txt', 'test.txt']
    
    # 为每个子集创建输出目录（images 存原图，masks 存分割掩码）
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, 'masks'), exist_ok=True)
    
    # 针对每个子集读取对应文件
    for subset_file in subset_files:
        subset_name = os.path.splitext(subset_file)[0]  # 得到 'train'、'val' 或 'test'
        txt_path = os.path.join(imgset_dir, subset_file)
        if not os.path.exists(txt_path):
            print(f"警告：找不到文件 {txt_path}")
            continue
        
        # 读取每个 txt 文件中存放的图片名称（不含后缀）
        with open(txt_path, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        
        print(f"开始处理子集：{subset_name}，共计 {len(names)} 个样本")
        
        # 逐个拷贝对应文件
        for name in names:
            # 原图文件，假定其扩展名为 .jpg
            image_file = os.path.join(images_src, name + '.jpg')
            if not os.path.exists(image_file):
                print(f"警告：找不到原图 {image_file}")
                continue

            # 掩码文件，假定其扩展名为 .png
            mask_file = os.path.join(masks_src, name + '.png')
            if not os.path.exists(mask_file):
                print(f"警告：找不到掩码图 {mask_file}")
                continue
            
            # 设定目标路径
            dst_image = os.path.join(output_dir, subset_name, 'images', name + '.jpg')
            dst_mask  = os.path.join(output_dir, subset_name, 'masks', name + '.png')
            
            # 复制文件
            shutil.copy(image_file, dst_image)
            shutil.copy(mask_file, dst_mask)
        
        print(f"子集 {subset_name} 处理完毕。")
    
    print("所有子集处理完成。")

if __name__ == '__main__':
    # VOC数据集的根目录
    voc_dataset_dir = '/dataset/zhuluoji/unet/unet_double/VOCdevkit3/VOC2007'
    # 设定提取后的数据存放目录，你可以根据需要修改
    output_directory = '/dataset/zhuluoji/unet/unet_double/VOCdevkit3/VOC2007_extracted'
    extract_images_and_masks(voc_dataset_dir, output_directory)