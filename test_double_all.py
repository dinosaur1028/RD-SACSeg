#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
from PIL import Image
from unet_double import Unet
from DexiNed_master.dexined_predict import predict_single_image as dexi_single
import cv2
import numpy as np
import torch
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
from skimage import exposure

def process_lab_a_channel(img):
    """
    对输入图像的LAB颜色空间中的A通道进行处理：
    1. 将图像转换为cv2可处理的格式，若输入为PIL.Image则先转换为BGR格式；
    2. 将BGR图像转换为LAB颜色空间，并取出A通道；
    3. 使用自适应直方图均衡化（CLAHE）增强对比度；
    4. 对增强后的通道进行2%和98%分位数的对比度拉伸（可近似0.5%截断效果）。
    
    返回处理后的A通道图像（PIL.Image格式的灰度图，即"L"模式）。
    """
    
    # 如果输入为 PIL.Image 格式，先转换为 numpy 数组，再转换为 BGR 格式
    if isinstance(img, Image.Image):
        # PIL图像默认是RGB格式
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 检查图像是否加载成功
    if img is None:
        raise ValueError("加载图像失败，请检查输入！")
    
    # 转换为LAB颜色空间，并提取A通道（OpenCV中转换后顺序为 L, A, B）
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_a = lab_img[:, :, 1]
    
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq_channel = clahe.apply(lab_a)
    
    # 对比度扩展：使用2%与98%分位数拉伸（可根据需要调整截断百分比）
    p2, p98 = np.percentile(eq_channel, (2, 98))
    stretched = exposure.rescale_intensity(eq_channel, in_range=(p2, p98))
    
    # exposure.rescale_intensity返回浮点数数据，
    # 如果数据值在[0,1]之间，则乘以255转换为uint8，否则直接转换为uint8
    if stretched.max() <= 1.0:
        stretched = (stretched * 255).astype(np.uint8)
    else:
        stretched = stretched.astype(np.uint8)
    
    # 将处理好的A通道转换为 PIL.Image 格式，并转换为灰度图（"L"模式）
    result_pil = Image.fromarray(stretched).convert("L")
    return result_pil

def dexined_process(orig_img):
    # 如果传入的图片是 PIL 格式，则将其转换为 numpy 格式，并转换为 BGR（与 cv2.imread 一致）
    if isinstance(orig_img, Image.Image):
        orig_img = np.array(orig_img)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    
    if orig_img is None:
        raise ValueError("加载图像失败，请检查路径！")
    
    # 设置模型预处理尺寸（例如模型要求尺寸为512x512）
    MODEL_IMG_WIDTH  = 512
    MODEL_IMG_HEIGHT = 512
    
    # 记录原始尺寸
    orig_height, orig_width = orig_img.shape[:2]
    
    # 为满足模型输入要求，将原图 resize 到模型尺寸
    resized_img = cv2.resize(orig_img, (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT))
    # 转换为 float32 类型
    resized_img = resized_img.astype(np.float32)
    # 转换为 [3, H, W] 格式（保持 BGR 顺序，与模型预处理要求一致）
    resized_img = np.transpose(resized_img, (2, 0, 1))
    
    # 调用 predict_single_image 得到预测结果，输出为 torch.Tensor，形状为 [H, W]，值域在 [0,1]
    prediction_tensor = dexi_single(resized_img)
    
    # 将预测结果 tensor 转换为 numpy 数组
    prediction_np = prediction_tensor.cpu().numpy()
    
    # 预测结果尺寸为模型尺寸 (MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH)
    # 恢复到原始图片尺寸
    prediction_restored = cv2.resize(prediction_np, (orig_width, orig_height))
    
    # 将值域从 [0,1] 映射到 [0,255]，并转换为 uint8 类型
    result_uint8 = (prediction_restored * 255).clip(0, 255).astype(np.uint8)
    
    # 如果你需要保存文件，可使用 cv2.imwrite 或 PIL.Image.save，这里示例使用 cv2.imwrite
    save_path = "predicted_result_restored.png"
    cv2.imwrite(save_path, result_uint8)
    print(f"预测结果已恢复到原始尺寸，并保存至 {save_path}")
    
    # 将 numpy 数组转换为 PIL.Image，并转换为灰度图（"L"模式）
    pil_result = Image.fromarray(result_uint8).convert("L")
    return pil_result


def predict_single_image(unet, img_path, out_r1_dir, out_r2_dir):
    """
    对单张图片进行预测，
    img_path   : 原始RGB图片路径（在JPEGImages文件夹下）
    out_r1_dir : 保存第一张预测结果的文件夹路径
    out_r2_dir : 保存第二张预测结果的文件夹路径
    """

    # 构造保存路径，使用原始文件名
    file_name = os.path.basename(img_path)
    r1_path = os.path.join(out_r1_dir, file_name)
    r2_path = os.path.join(out_r2_dir, file_name)

    # 判断图片是否存在
    if not os.path.exists(img_path):
        print(f"原图不存在：{img_path}")
        return


    try:
        image = Image.open(img_path)
        # dex   = Image.open(dex_path).convert("L")
        dex = dexined_process(image)
        # lab   = Image.open(lab_path).convert("L")
        lab = process_lab_a_channel(image)
    except Exception as e:
        print(f"读取图片出错：{img_path} 或对应的 dex/lab 图片，错误信息：{e}")
        return

    # 调用模型进行预测，返回两个结果图片
    result_image1, result_image2 = unet.detect_image(image, dex, lab)

    

    result_image1.save(r1_path)
    result_image2.save(r2_path)
    print(f"图片 {file_name} 预测完成，已保存到：\n {r1_path}\n {r2_path}")
    return result_image1

def batch_predict(unet, input_dir, output_root):
    """
    unet       : 模型对象
    input_dir  : 待预测图片所在文件夹路径（要求图片存放在此文件夹中的 JPEGImages 文件夹下）
    output_root: 保存结果的根目录，结果分别保存在 output_root/r1、output_root/r2
    """
    if not os.path.exists(input_dir):
        print("输入文件夹不存在，请检查路径。")
        return

    # 创建输出文件夹，其中包含两个子文件夹用于保存两种预测结果
    os.makedirs(output_root, exist_ok=True)
    out_r1_dir = os.path.join(output_root, "r1")
    out_r2_dir = os.path.join(output_root, "r2")
    os.makedirs(out_r1_dir, exist_ok=True)
    os.makedirs(out_r2_dir, exist_ok=True)

    # 假设所有待预测图片均存储在 input_dir/JPEGImages 下
    jpeg_dir = input_dir
    if not os.path.exists(jpeg_dir):
        print(f"找不到 JPEGImages 文件夹：{jpeg_dir}")
        return
    print(jpeg_dir)

    # 仅处理常见图像格式
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    image_list = [os.path.join(jpeg_dir, f) for f in os.listdir(jpeg_dir)
                  if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_list:
        print("没有找到符合条件的图片文件。")
        return

    print(f"检测到 {len(image_list)} 张图片，将开始批量预测...")
    for img_path in image_list:
        predict_single_image(unet, img_path, out_r1_dir, out_r2_dir)
    print("批量预测完成。")

def main():
    # 初始化模型
    unet = Unet()
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    # # 单张图片预测（原有使用方式）
    # img_path = "/dataset/zhuluoji/unet/unet_double/VOCdevkit3/VOC2007/JPEGImages/1_left_front.jpg"
    # dex_path = img_path.replace("JPEGImages", "mask_output_folder")
    # lab_path = img_path.replace("JPEGImages", "LAB_dir")
    # dex = Image.open(dex_path).convert("L")
    # lab = Image.open(lab_path).convert("L")
    # save_dir = f"/dataset/zhuluoji/unet/unet_double/single_img_output/{timestr}"
    # os.mkdir(save_dir)
    # r1_dir = os.path.join(save_dir, "r1")
    # r2_dir = os.path.join(save_dir, "r2")
    # os.mkdir(r1_dir)
    # os.mkdir(r2_dir)
    # r1_path = os.path.join(r1_dir, os.path.basename(img_path))
    # r2_path = os.path.join(r2_dir, os.path.basename(img_path))

    # try:
    #     image = Image.open(img_path)
    # except Exception as e:
    #     print("打开图片出错：", e)
    #     return

    # result_image1, result_image2 = unet.detect_image(image, dex, lab)
    # result_image1.save(r1_path)
    # result_image2.save(r2_path)
    # print(f"单张图片预测结果已保存到：\n{r1_path}\n{r2_path}")

    # 批量预测：设定待预测图片根目录（该目录下需要包含 JPEGImages 文件夹，
    # 同时根据 JPEGImages 路径能对应到 dex（mask_output_folder）和 lab（LAB_dir）图片）
    input_folder = "/dataset/zhuluoji/unet/unet_double/img_test_20260320" 
    batch_out_dir = f"/dataset/zhuluoji/unet/unet_double/batch_img_output/{timestr}"
    batch_predict(unet, input_folder, batch_out_dir)

if __name__ == '__main__':
    main()