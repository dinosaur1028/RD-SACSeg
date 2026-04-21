#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
from PIL import Image
from unet import Unet

def predict_single_image(unet, img_path, out_r1_dir, out_r2_dir):
    try:
        image = Image.open(img_path)
    except Exception as e:
        print(f"打开图片 {img_path} 出错：", e)
        return

    # 调用模型的detect_image2接口进行预测，返回两个结果图片
    result_image1, result_image2 = unet.detect_image2(image)
    # 构造保存路径（使用原始文件名）
    file_name = os.path.basename(img_path)
    r1_path = os.path.join(out_r1_dir, file_name)
    r2_path = os.path.join(out_r2_dir, file_name)
    result_image1.save(r1_path)
    result_image2.save(r2_path)
    print(f"图片 {file_name} 预测完成，已保存到：\n {r1_path}\n {r2_path}")

def batch_predict(unet, input_dir, output_root):
    """
    unet       : 预测模型对象
    input_dir  : 待预测的图片所在文件夹路径
    output_root: 保存结果的文件夹路径（会自动创建子文件夹 r1 和 r2）
    """
    if not os.path.exists(input_dir):
        print("输入文件夹不存在，请检查路径。")
        return

    # 创建输出文件夹，包含两个子文件夹，用于保存两种预测结果
    os.makedirs(output_root, exist_ok=True)
    out_r1_dir = os.path.join(output_root, "r1")
    out_r2_dir = os.path.join(output_root, "r2")
    os.makedirs(out_r1_dir, exist_ok=True)
    os.makedirs(out_r2_dir, exist_ok=True)

    # 遍历文件夹下所有图片，按扩展名过滤（可根据需要扩展）
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    image_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_list:
        print("没有找到符合条件的图片文件。")
        return

    print(f"检测到 {len(image_list)} 张图片，将开始批量预测...")
    # 遍历所有图片，依次预测保存
    for img_path in image_list:
        predict_single_image(unet, img_path, out_r1_dir, out_r2_dir)
    print("批量预测完成。")


def main():
    # 初始化模型
    unet = Unet()
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    # # 单张图片预测（原有使用方式）
    # img_path = "/dataset/zhuluoji/unet/unet_double/VOCdevkit3/VOC2007/JPEGImages/1_left_front.jpg"
    
    # single_out_dir = f"/dataset/zhuluoji/unet/unet_double/single_img_output/{timestr}"
    # os.makedirs(single_out_dir, exist_ok=True)
    # r1_dir = os.path.join(single_out_dir, 'r1')
    # r2_dir = os.path.join(single_out_dir, 'r2')
    # os.makedirs(r1_dir, exist_ok=True)
    # os.makedirs(r2_dir, exist_ok=True)
    
    # if not os.path.exists(img_path):
    #     print("图片文件不存在，请检查路径。")
    #     return

    # try:
    #     image = Image.open(img_path)
    # except Exception as e:
    #     print("打开图片出错：", e)
    #     return

    # result_image1, result_image2 = unet.detect_image2(image)
    # result_image1.save(os.path.join(r1_dir, os.path.basename(img_path)))
    # result_image2.save(os.path.join(r2_dir, os.path.basename(img_path)))
    # print(f"单张图片预测结果已保存！")

    # 批量预测
    # 修改 input_folder 为待预测图片文件夹路径，output_folder 为保存结果的根目录，
    # 结果将会分别保存在 output_folder/r1 和 output_folder/r2 文件夹内
    input_folder = "/dataset/zhuluoji/unet/unet_double/img_new"
    batch_out_dir = f"/dataset/zhuluoji/unet/unet_double/batch_img_output/{timestr}"
    batch_predict(unet, input_folder, batch_out_dir)


if __name__ == '__main__':
    main()