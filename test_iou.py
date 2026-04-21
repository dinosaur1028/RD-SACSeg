#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型测试及指标计算示例代码（统计整个测试集的平均指标）
"""

import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
from test_double_all import predict_single_image
from unet_double import Unet
from PIL import Image
import datetime
import glob

def visualize_instance_comparisons(gt_mask, pred_mask, output_dir, 
                                  image_base_name, orig_image=None, margin=10):
    """
    可视化每个实例的真实掩码、预测掩码及其重叠区域
    参数:
      gt_mask -- 真实分割掩码 (H, W)，值为0(背景)和1(前景)
      pred_mask -- 预测分割掩码 (H, W)，值为0(背景)和1(前景)
      output_dir -- 输出图像保存目录
      image_base_name -- 基础文件名(用于保存结果)
      orig_image -- 原始RGB图像 (可选)
      margin -- 边界扩展像素数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    labeled_gt = label(gt_mask)
    regions = regionprops(labeled_gt)
    processed_instances = 0
    
    for i, region in enumerate(regions):
        try:
            minr, minc, maxr, maxc = region.bbox
            minr = max(0, minr - margin)
            minc = max(0, minc - margin)
            maxr = min(gt_mask.shape[0], maxr + margin)
            maxc = min(gt_mask.shape[1], maxc + margin)
            
            if maxr <= minr or maxc <= minc:
                print(f"警告: 实例 {i+1} 的边界框无效 ({minr}:{maxr}, {minc}:{maxc})，跳过")
                continue
            
            gt_roi = gt_mask[minr:maxr, minc:maxc]
            if minr >= pred_mask.shape[0] or minc >= pred_mask.shape[1] or maxr > pred_mask.shape[0] or maxc > pred_mask.shape[1]:
                print(f"警告: 实例 {i+1} 的边界框超出预测掩码范围 ({minr}:{maxr}, {minc}:{maxc}) vs ({pred_mask.shape[0]}, {pred_mask.shape[1]})，跳过")
                continue
            
            pred_roi = pred_mask[minr:maxr, minc:maxc]
            if gt_roi.shape != pred_roi.shape:
                print(f"警告: 实例 {i+1} 的ROI形状不匹配 (gt: {gt_roi.shape}, pred: {pred_roi.shape})，跳过")
                continue
            
            if orig_image is not None:
                if minr < orig_image.shape[0] and minc < orig_image.shape[1] and maxr <= orig_image.shape[0] and maxc <= orig_image.shape[1]:
                    visual_img = orig_image[minr:maxr, minc:maxc].copy()
                else:
                    print(f"警告: 实例 {i+1} 的边界框超出原图范围，使用黑色背景")
                    visual_img = np.zeros((gt_roi.shape[0], gt_roi.shape[1], 3), dtype=np.uint8)
            else:
                visual_img = np.zeros((gt_roi.shape[0], gt_roi.shape[1], 3), dtype=np.uint8)
            
            # 保存 ground truth 单独叠加图
            gt_only_img = visual_img.copy()
            gt_mask_area = gt_roi > 0
            green_color = [0, 255, 0]
            alpha = 0.7
            gt_only_img = gt_only_img.astype(np.float32)
            gt_only_img[gt_mask_area] = gt_only_img[gt_mask_area] * (1 - alpha) + np.array(green_color) * alpha
            gt_only_img = np.clip(gt_only_img, 0, 255).astype(np.uint8)
            gt_output_path = os.path.join(output_dir, f"{image_base_name}_gt_{i+1}.png")
            cv2.imwrite(gt_output_path, cv2.cvtColor(gt_only_img, cv2.COLOR_RGB2BGR))
            
            overlap = np.logical_and(gt_roi, pred_roi)
            only_gt = np.logical_and(gt_roi, np.logical_not(pred_roi))
            only_pred = np.logical_and(pred_roi, np.logical_not(gt_roi))
            
            green_color_arr = np.array([0, 200, 0], dtype=np.uint8)
            visual_img[only_gt] = (visual_img[only_gt].astype(np.float32) * (1 - alpha) + green_color_arr * alpha).astype(np.uint8)
            
            red_color = np.array([200, 0, 0], dtype=np.uint8)
            visual_img[only_pred] = (visual_img[only_pred].astype(np.float32) * (1 - alpha) + red_color * alpha).astype(np.uint8)
            
            yellow_color = np.array([255, 255, 0], dtype=np.uint8)
            visual_img[overlap] = yellow_color
            
            cv2.rectangle(visual_img, (0, 0), (visual_img.shape[1]-1, visual_img.shape[0]-1), (255, 255, 255), 2)
            output_path = os.path.join(output_dir, f"{image_base_name}_instance_{i+1}.png")
            cv2.imwrite(output_path, cv2.cvtColor(visual_img, cv2.COLOR_RGB2BGR))
            processed_instances += 1
            
        except Exception as e:
            print(f"处理实例 {i+1} 时出错: {str(e)}")
    
    print(f"保存了 {processed_instances}/{len(regions)} 个实例可视化结果到 {output_dir}")


def compute_metrics(gt_mask, pred_mask):
    """
    计算分割指标：IoU、Precision 和 Dice系数
    参数:
      gt_mask: 真实mask, 二值图（0与1）
      pred_mask: 预测mask, 二值图（0与1）
    返回:
      iou, precision, dice
    """
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = intersection / union if union > 0 else 1.0
    pred_sum = pred_mask.sum()
    precision = intersection / pred_sum if pred_sum > 0 else 1.0
    gt_sum = gt_mask.sum()
    dice = (2 * intersection) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 1.0
    return iou, precision, dice


def process_single_image(image_path, output_dir, unet):
    """
    处理单张图像，计算并返回指标 (iou, precision, dice)
    """
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    base_name = "comparison"
    
    orig_image_path = image_path
    gt_mask_path = f"/dataset/zhuluoji/unet/unet_double/VOCdevkit_dilate/VOC2007/test_mask/{img_name}.png"
    
    out_r1_dir = os.path.join(output_dir, "r1")
    out_r2_dir = os.path.join(output_dir, "r2")
    os.makedirs(out_r1_dir, exist_ok=True)
    os.makedirs(out_r2_dir, exist_ok=True)
    
    print(f"处理图像: {img_name}")
    
    orig_img = cv2.imread(orig_image_path)
    if orig_img is None:
        print(f"错误: 无法读取图像 {orig_image_path}")
        return None
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"错误: 无法读取mask {gt_mask_path}")
        return None
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    pred_mask_pil = predict_single_image(unet, orig_image_path, out_r1_dir, out_r2_dir)
    pred_mask_np = np.array(pred_mask_pil)
    
    if len(pred_mask_np.shape) == 3:
        pred_mask_gray = np.mean(pred_mask_np, axis=2).astype(np.uint8)
    else:
        pred_mask_gray = pred_mask_np
    
    pred_mask = (pred_mask_gray > 10).astype(np.uint8)
    if pred_mask.shape != orig_img.shape[:2]:
        print(f"调整预测掩码尺寸: {pred_mask.shape} -> {orig_img.shape[:2]}")
        try:
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"调整预测掩码尺寸失败，跳过该图像: {str(e)}")
            return None
    
    # 计算图像整体的分割指标
    iou, precision, dice = compute_metrics(gt_mask, pred_mask)
    print(f"[{img_name}] mIoU: {iou:.4f}, Precision: {precision:.4f}, Dice: {dice:.4f}")
    
    # 将指标写入文本文件（可选）
    metrics_txt = os.path.join(output_dir, f"{img_name}_metrics.txt")
    with open(metrics_txt, "w") as f:
        f.write(f"mIoU: {iou:.4f}\nPrecision: {precision:.4f}\nDice: {dice:.4f}\n")
    
    visualize_instance_comparisons(
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        output_dir=output_dir,
        image_base_name=base_name,
        orig_image=orig_img,
        margin=15
    )
    
    full_gt_overlay = orig_img.copy()
    green_color = [0, 255, 0]
    alpha = 0.7
    full_gt_overlay = full_gt_overlay.astype(np.float32)
    gt_mask_area = gt_mask.astype(bool)
    full_gt_overlay[gt_mask_area] = full_gt_overlay[gt_mask_area] * (1 - alpha) + np.array(green_color) * alpha
    full_gt_overlay = np.clip(full_gt_overlay, 0, 255).astype(np.uint8)
    
    full_gt_path = os.path.join(output_dir, f"{base_name}_full_gt.png")
    cv2.imwrite(full_gt_path, cv2.cvtColor(full_gt_overlay, cv2.COLOR_RGB2BGR))
    print(f"保存整张 ground truth 叠加图到: {full_gt_path}")
    
    # 返回当前图像的指标
    return iou, precision, dice


if __name__ == "__main__":
    unet = Unet()
    
    process_mode = "folder"  # 可选 "single" 或 "folder"
    
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_directory = f"/dataset/zhuluoji/unet/unet_double/output_new_test/{time_str}"
    os.makedirs(output_directory, exist_ok=True)
    
    if process_mode == "single":
        image_path = "/dataset/zhuluoji/unet/unet_double/VOCdevkit3_copy/VOC2007/JPEGImages/6_right_front.jpg"
        metrics = process_single_image(image_path, output_directory, unet)
        if metrics is not None:
            iou, precision, dice = metrics
            print(f"单张图指标：mIoU={iou:.4f}, Precision={precision:.4f}, Dice={dice:.4f}")
    
    elif process_mode == "folder":
        folder_path = "/dataset/zhuluoji/unet/unet_double/VOCdevkit_dilate/VOC2007/test_origin"
        image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
        print(f"找到 {len(image_paths)} 张图片待处理")
        
        # 累加指标
        total_iou = 0.0
        total_precision = 0.0
        total_dice = 0.0
        valid_image_count = 0
        
        for i, img_path in enumerate(image_paths):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_output_dir = os.path.join(output_directory, img_name)
            os.makedirs(img_output_dir, exist_ok=True)
            
            print(f"\n处理进度: {i+1}/{len(image_paths)} - {img_name}")
            try:
                metrics = process_single_image(img_path, img_output_dir, unet)
                if metrics is not None:
                    iou, precision, dice = metrics
                    total_iou += iou
                    total_precision += precision
                    total_dice += dice
                    valid_image_count += 1
            except Exception as e:
                print(f"处理图像 {img_name} 时出错: {str(e)}")
                if os.path.exists(img_output_dir) and not os.listdir(img_output_dir):
                    os.rmdir(img_output_dir)
        
        if valid_image_count > 0:
            avg_iou = total_iou / valid_image_count
            avg_precision = total_precision / valid_image_count
            avg_dice = total_dice / valid_image_count
            print("\n整个测试集的平均指标：")
            print(f"平均 mIoU: {avg_iou:.4f}")
            print(f"平均 Precision: {avg_precision:.4f}")
            print(f"平均 Dice: {avg_dice:.4f}")
            
            # 可选：将整体指标保存成txt文件
            summary_txt = os.path.join(output_directory, "overall_metrics.txt")
            with open(summary_txt, "w") as f:
                f.write(f"处理图像数: {valid_image_count}\n")
                f.write(f"平均 mIoU: {avg_iou:.4f}\n")
                f.write(f"平均 Precision: {avg_precision:.4f}\n")
                f.write(f"平均 Dice: {avg_dice:.4f}\n")
            print(f"整体指标保存至：{summary_txt}")
        else:
            print("没有有效的图像指标可供计算平均值。")
    
    else:
        print(f"错误: 未知的处理模式 '{process_mode}'. 请使用 'single' 或 'folder'.")
