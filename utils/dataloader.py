#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path,
                 use_dex=True, use_lab=True):
        """
        annotation_lines: 每行数据，第一项为图像名称（不含后缀）
        input_shape: 输入尺寸 [h, w]
        num_classes: 分割类别数（真实类别数，不包含背景，此处后面会+1）
        train: 是否处于训练模式（会影响数据增强）
        dataset_path: 数据集根目录，即VOCdevkit所在路径
        use_dex: 是否使用 Dexined_Aug 文件夹中的灰度图，
        use_lab: 是否使用 LAB_Aug 文件夹中的灰度图，
                 如果为 False，则用全0图像填充对应通道
        """
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length           = len(annotation_lines)
        self.input_shape      = input_shape
        self.num_classes      = num_classes
        self.train            = train
        self.dataset_path     = dataset_path
        self.use_dex          = use_dex
        self.use_lab          = use_lab

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   加载增强后的图像及对应的多个 mask
        #   jpg：RGB图像
        #   png：SegmentationClass_Aug 中的 mask（后续 one-hot 编码使用）
        #   dex：Dexined_Aug 中的灰度图，处理后扩展为 (h, w, 1)
        #   lab：LAB_Aug 中的灰度图，处理后扩展为 (h, w, 1)
        #       当 use_dex/use_lab 为False时，则用全0图像填充对应通道
        #-------------------------------#
        jpg_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages_Aug", name + ".jpg")
        png_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass_Aug", name + ".png")
        try:
            jpg   = Image.open(jpg_path)
        except:
            print(1)
        png   = Image.open(png_path)

        if self.use_dex:
            dex_path = os.path.join(self.dataset_path, "VOC2007/Dexined_Aug", name + ".png")
            dex   = Image.open(dex_path).convert("L")  # 保证为灰度图
        else:
            dex   = Image.new("L", jpg.size, 0)

        if self.use_lab:
            lab_path = os.path.join(self.dataset_path, "VOC2007/LAB_Aug", name + ".png")
            lab   = Image.open(lab_path).convert("L")  # 保证为灰度图
        else:
            lab   = Image.new("L", jpg.size, 0)

        #-------------------------------#
        #   数据增强（几何变换等），保证 jpg、png、dex、lab 同步变换
        #-------------------------------#
        # 修改 get_random_data 方法，加入 lab 图像参数，random参数由 self.train 决定
        jpg, png, dex, lab = self.get_random_data(jpg, png, dex, lab, self.input_shape, random=self.train)

        #-------------------------------------#
        #  对处理后的 jpg 进行预处理，并拼接 dex、lab 通道
        #-------------------------------------#
        # 将 PIL.Image 转为 numpy 数组（RGB→float）
        jpg_array  = np.array(jpg, np.float64)
        # 对 RGB 图像预处理（例如归一化、减均值等）
        jpg_array  = preprocess_input(jpg_array)
        # jpg_array (h, w, 3)，将 dex、lab 转为 numpy 数组并扩展维度 (h, w, 1)
        dex_array  = np.array(dex, np.float64)[..., np.newaxis]
        lab_array  = np.array(lab, np.float64)[..., np.newaxis]
        # 根据需要对 dex 和 lab 归一化（例如归一化到0~1）
        dex_array  = dex_array / 255.0
        lab_array  = lab_array / 255.0
        # 拼接为一个 5 通道的输入：RGB + dex + lab
        combo_jpg  = np.concatenate([jpg_array, dex_array, lab_array], axis=-1)
        # 转换为 (C, H, W)
        combo_jpg  = np.transpose(combo_jpg, [2, 0, 1])

        #-------------------------------------#
        #   对 png 进行处理与 one-hot 编码（与原代码保持一致）, png 为分割标签
        #-------------------------------------#
        png_array = np.array(png)
        png_array[png_array >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[png_array.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return combo_jpg, png_array, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, dex, lab, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        """
        数据增强：
          image: 原 RGB 图像（PIL.Image）
          label: 对应的 mask（PIL.Image，模式为 L）
          dex  : 对应的 Dexined 灰度图（PIL.Image，模式为 L）
          lab  : 对应的 LAB 灰度图（PIL.Image，模式为 L）
        当 random = False 时，仅作 resize 与填充处理；
        当 random = True 时，作缩放、翻转、随机平移以及色域变换（仅对 image 有效），
        label、dex、lab 三者同步变换。
        """
        # 保证 image 为 RGB 格式
        image = cvtColor(image)
        # label、dex、lab 均转换为 PIL.Image 格式
        label = Image.fromarray(np.array(label))
        dex   = Image.fromarray(np.array(dex))
        lab   = Image.fromarray(np.array(lab))

        iw, ih = image.size
        h, w   = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw    = int(iw * scale)
            nh    = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', (w, h), (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))

            dex = dex.resize((nw, nh), Image.NEAREST)
            new_dex = Image.new('L', (w, h), (0))
            new_dex.paste(dex, ((w - nw) // 2, (h - nh) // 2))

            lab = lab.resize((nw, nh), Image.NEAREST)
            new_lab = Image.new('L', (w, h), (0))
            new_lab.paste(lab, ((w - nw) // 2, (h - nh) // 2))

            return new_image, new_label, new_dex, new_lab

        # 随机缩放与长宽扭曲
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        dex   = dex.resize((nw, nh), Image.NEAREST)
        lab   = lab.resize((nw, nh), Image.NEAREST)

        # 随机翻转
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            dex   = dex.transpose(Image.FLIP_LEFT_RIGHT)
            lab   = lab.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机平移
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        new_label = Image.new('L', (w, h), (0))
        new_label.paste(label, (dx, dy))
        new_dex   = Image.new('L', (w, h), (0))
        new_dex.paste(dex, (dx, dy))
        new_lab   = Image.new('L', (w, h), (0))
        new_lab.paste(lab, (dx, dy))
        image = new_image
        label = new_label
        dex   = new_dex
        lab   = new_lab

        # 色域变换：仅对 image 有效
        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hsv_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv_image)
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        h_channel = cv2.LUT(h_channel, lut_hue)
        s_channel = cv2.LUT(s_channel, lut_sat)
        v_channel = cv2.LUT(v_channel, lut_val)
        hsv_image = cv2.merge((h_channel, s_channel, v_channel))
        image_data = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        image = Image.fromarray(image_data)

        return image, label, dex, lab


# DataLoader中 collate_fn 使用（保持不变）
def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images     = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs       = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels