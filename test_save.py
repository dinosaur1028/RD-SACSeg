import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
from test_double_all import predict_single_image
from unet_double import Unet
from PIL import Image
import datetime
import glob
import time
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import json

def calculate_metrics(gt_mask, pred_mask):
    """
    计算多种分割评估指标
    
    参数:
    gt_mask -- 真实掩码，二值图(0或1)
    pred_mask -- 预测掩码，二值图(0或1)
    
    返回:
    dict -- 包含多种评估指标的字典
    """
    # 确保掩码是二值的
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    
    # 将布尔掩码扁平化为1D数组
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()
    
    # 计算基本像素统计量
    tp = np.sum(np.logical_and(pred_flat, gt_flat))
    fp = np.sum(np.logical_and(pred_flat, np.logical_not(gt_flat)))
    fn = np.sum(np.logical_and(np.logical_not(pred_flat), gt_flat))
    tn = np.sum(np.logical_and(np.logical_not(pred_flat), np.logical_not(gt_flat)))
    
    # 计算评估指标
    metrics = {}
    
    # 准确率 (Accuracy)
    metrics['accuracy'] = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    
    # 精确率 (Precision)
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # 召回率 (Recall / Sensitivity)
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 特异度 (Specificity)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # F1分数
    metrics['f1_score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    # IoU (Intersection over Union) / Jaccard指数
    metrics['iou'] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # Dice系数 (与F1分数相同)
    metrics['dice'] = metrics['f1_score']
    
    # 使用sklearn计算（包含背景类）
    try:
        # 使用sklearn计算包含背景类的指标
        classes = [0, 1]  # 0=背景, 1=前景
        
        # 计算每个类别的指标
        precision_per_class = precision_score(gt_flat, pred_flat, labels=classes, average=None, zero_division=0)
        recall_per_class = recall_score(gt_flat, pred_flat, labels=classes, average=None, zero_division=0)
        f1_per_class = f1_score(gt_flat, pred_flat, labels=classes, average=None, zero_division=0)
        iou_per_class = jaccard_score(gt_flat, pred_flat, labels=classes, average=None, zero_division=0)
        
        # 保存每个类别的指标
        metrics['precision_bg'] = float(precision_per_class[0])
        metrics['precision_fg'] = float(precision_per_class[1])
        metrics['recall_bg'] = float(recall_per_class[0])
        metrics['recall_fg'] = float(recall_per_class[1])
        metrics['f1_bg'] = float(f1_per_class[0])
        metrics['f1_fg'] = float(f1_per_class[1])
        metrics['iou_bg'] = float(iou_per_class[0])
        metrics['iou_fg'] = float(iou_per_class[1])
        
        # 计算平均指标（包括背景类）
        metrics['mean_precision'] = float(np.mean(precision_per_class))
        metrics['mean_recall'] = float(np.mean(recall_per_class))
        metrics['mean_f1'] = float(np.mean(f1_per_class))
        metrics['mean_iou'] = float(np.mean(iou_per_class))
        
    except Exception as e:
        print(f"计算sklearn指标时出错: {str(e)}")
        metrics['mean_precision'] = 0
        metrics['mean_recall'] = 0
        metrics['mean_f1'] = 0
        metrics['mean_iou'] = 0
    
    return metrics

def visualize_instance_comparisons(gt_mask, pred_mask, output_dir, 
                                  image_base_name, orig_image=None, margin=10):
    """
    可视化每个实例的真实掩码、预测掩码及其重叠区域
    参数:
    gt_mask -- 真实分割掩码 (H, W) 值为0(背景)和1(前景)
    pred_mask -- 预测分割掩码 (H, W) 值为0(背景)和1(前景)
    output_dir -- 输出图像保存目录
    image_base_name -- 基础文件名(用于保存结果)
    orig_image -- 原始RGB图像 (可选)
    margin -- 边界扩展像素数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 连通域分析获取真实实例
    labeled_gt = label(gt_mask)
    regions = regionprops(labeled_gt)
    
    # 记录处理的实例数
    processed_instances = 0
    
    for i, region in enumerate(regions):
        try:
            # 获取最小边界框并扩展
            minr, minc, maxr, maxc = region.bbox
            minr = max(0, minr - margin)
            minc = max(0, minc - margin)
            maxr = min(gt_mask.shape[0], maxr + margin)
            maxc = min(gt_mask.shape[1], maxc + margin)
            
            # 确保边界框有效
            if maxr <= minr or maxc <= minc:
                print(f"警告: 实例 {i+1} 的边界框无效 ({minr}:{maxr}, {minc}:{maxc})，跳过")
                continue
            
            # 截取感兴趣区域(ROI)
            gt_roi = gt_mask[minr:maxr, minc:maxc]
            
            # 确保预测掩码有足够大小
            if minr >= pred_mask.shape[0] or minc >= pred_mask.shape[1] or maxr > pred_mask.shape[0] or maxc > pred_mask.shape[1]:
                print(f"警告: 实例 {i+1} 的边界框超出预测掩码范围 ({minr}:{maxr}, {minc}:{maxc}) vs ({pred_mask.shape[0]}, {pred_mask.shape[1]})，跳过")
                continue
            
            pred_roi = pred_mask[minr:maxr, minc:maxc]
            
            # 确保ROI大小匹配
            if gt_roi.shape != pred_roi.shape:
                print(f"警告: 实例 {i+1} 的ROI形状不匹配 (gt: {gt_roi.shape}, pred: {pred_roi.shape})，跳过")
                continue
            
            # 创建可视化图像
            if orig_image is not None:
                # 确保边界框在原图范围内
                if minr < orig_image.shape[0] and minc < orig_image.shape[1] and maxr <= orig_image.shape[0] and maxc <= orig_image.shape[1]:
                    visual_img = orig_image[minr:maxr, minc:maxc].copy()
                else:
                    print(f"警告: 实例 {i+1} 的边界框超出原图范围，使用黑色背景")
                    visual_img = np.zeros((gt_roi.shape[0], gt_roi.shape[1], 3), dtype=np.uint8)
            else:
                visual_img = np.zeros((gt_roi.shape[0], gt_roi.shape[1], 3), dtype=np.uint8)
            
            # ======== 保存ground truth单独叠加图（裁剪版） ========
            gt_only_img = visual_img.copy()
            
            # 应用绿色半透明效果到ground truth区域
            gt_mask_area = gt_roi > 0  # 确保是布尔掩码
            
            # 使用更简单的颜色混合方法
            green_color = [0, 255, 0]  # 纯绿色
            alpha = 0.7  # 透明度
            
            # 将RGB值转换为浮点数进行计算
            gt_only_img = gt_only_img.astype(np.float32)
            gt_only_img[gt_mask_area] = gt_only_img[gt_mask_area] * (1 - alpha) + np.array(green_color) * alpha
            gt_only_img = np.clip(gt_only_img, 0, 255).astype(np.uint8)
            
            # 保存ground truth单独叠加图（裁剪版）
            gt_output_path = os.path.join(output_dir, f"{image_base_name}_gt_{i+1}.png")
            cv2.imwrite(gt_output_path, cv2.cvtColor(gt_only_img, cv2.COLOR_RGB2BGR))
            # ======== 新增代码结束 ========
            
            # 创建彩色掩码图层
            overlap = np.logical_and(gt_roi, pred_roi)
            only_gt = np.logical_and(gt_roi, np.logical_not(pred_roi))
            only_pred = np.logical_and(pred_roi, np.logical_not(gt_roi))
            
            # 修正颜色混合操作
            # 真实掩码 - 绿色（半透明）
            green_color_arr = np.array([0, 200, 0], dtype=np.uint8)
            alpha = 0.7
            visual_img[only_gt] = (visual_img[only_gt].astype(np.float32) * (1 - alpha) + 
                                   green_color_arr * alpha).astype(np.uint8)
            
            # 预测掩码 - 红色（半透明）
            red_color = np.array([200, 0, 0], dtype=np.uint8)
            visual_img[only_pred] = (visual_img[only_pred].astype(np.float32) * (1 - alpha) + 
                                     red_color * alpha).astype(np.uint8)
            
            # 重叠区域 - 黄色（实心）
            yellow_color = np.array([255, 255, 0], dtype=np.uint8)
            visual_img[overlap] = yellow_color
            
            # 添加边界框
            cv2.rectangle(visual_img, (0, 0), 
                         (visual_img.shape[1]-1, visual_img.shape[0]-1), 
                         (255, 255, 255), 2)
            
            # 保存结果
            output_path = os.path.join(output_dir, f"{image_base_name}_instance_{i+1}.png")
            cv2.imwrite(output_path, cv2.cvtColor(visual_img, cv2.COLOR_RGB2BGR))
            
            processed_instances += 1
            
        except Exception as e:
            print(f"处理实例 {i+1} 时出错: {str(e)}")
    
    print(f"保存了 {processed_instances}/{len(regions)} 个实例可视化结果到 {output_dir}")

def process_single_image(image_path, output_dir, unet, metrics_list):
    """
    处理单张图像的函数，并返回计算的评估指标
    """
    # 从文件路径中提取基本文件名（不含扩展名）
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    base_name = "comparison"
    
    # 配置路径
    orig_image_path = image_path
    gt_mask_path = f"/dataset/zhuluoji/unet/unet_double/VOCdevkit3_copy/VOC2007/SegmentationClass_dilated_2/{img_name}.png"
    
    # 创建输出子目录
    out_r1_dir = os.path.join(output_dir, "r1")
    out_r2_dir = os.path.join(output_dir, "r2")
    os.makedirs(out_r1_dir, exist_ok=True)
    os.makedirs(out_r2_dir, exist_ok=True)
    
    print(f"处理图像: {img_name}")
    
    # 读取原始图像和真实掩码
    orig_img = cv2.imread(orig_image_path)
    if orig_img is None:
        print(f"错误: 无法读取图像 {orig_image_path}")
        return None
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    
    # 读取真实掩码并二值化
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"错误: 无法读取mask {gt_mask_path}")
        return None
    gt_mask = (gt_mask > 0).astype(np.uint8)  # 二值化为0和1
    
    # 获取预测掩码（PIL Image对象）
    pred_mask_pil = predict_single_image(unet, orig_image_path, out_r1_dir, out_r2_dir)
    
    # 将PIL Image转换为numpy数组并进行灰度二值化
    pred_mask_np = np.array(pred_mask_pil)
    
        # 检查图像是RGB还是灰度
    if len(pred_mask_np.shape) == 3:  # 如果是RGB图像
        # 转换为灰度图（平均值）
        pred_mask_gray = np.mean(pred_mask_np, axis=2).astype(np.uint8)
    else:  # 如果是灰度图
        pred_mask_gray = pred_mask_np
    
    # 低阈值二值化：大于10的像素视为前景(1)，小于等于10的像素视为背景(0)
    pred_mask = (pred_mask_gray > 10).astype(np.uint8)
    
    # 确保预测掩码与原始图像尺寸相同
    if pred_mask.shape != orig_img.shape[:2]:
        print(f"调整预测掩码尺寸: {pred_mask.shape} -> {orig_img.shape[:2]}")
        try:
            # 使用最近邻插值保持二值特性
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                                  (orig_img.shape[1], orig_img.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        except:
            print("调整预测掩码尺寸失败，跳过该图像")
            return None
    
    # 计算评估指标
    metrics = calculate_metrics(gt_mask, pred_mask)
    metrics['image_name'] = img_name
    metrics_list.append(metrics)
    
    # 在控制台输出该图像的评估指标
    print(f"\n图像 {img_name} 的评估指标:")
    print(f"IoU (前景): {metrics['iou_fg']:.4f}")
    print(f"IoU (背景): {metrics['iou_bg']:.4f}")
    print(f"平均IoU: {metrics['mean_iou']:.4f}")
    print(f"Dice (前景): {metrics['f1_fg']:.4f}")
    print(f"平均Dice/F1: {metrics['mean_f1']:.4f}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    
    # 生成可视化结果（实例级别的）
    visualize_instance_comparisons(
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        output_dir=output_dir,
        image_base_name=base_name,
        orig_image=orig_img,
        margin=15  # 扩展边界15像素
    )
    
    # ======== 保存整张ground truth叠加图 ========
    # 创建整张ground truth叠加图的副本
    full_gt_overlay = orig_img.copy()
    
    # 应用绿色半透明效果到整个ground truth区域
    green_color = [0, 255, 0]  # 绿色
    alpha = 0.7  # 透明度
    
    # 将RGB值转换为浮点数进行计算
    full_gt_overlay = full_gt_overlay.astype(np.float32)
    gt_mask_area = gt_mask.astype(bool)  # 确保是布尔掩码
    
    # 使用矢量化操作提高效率
    full_gt_overlay[gt_mask_area] = (
        full_gt_overlay[gt_mask_area] * (1 - alpha) + 
        np.array(green_color) * alpha
    )
    full_gt_overlay = np.clip(full_gt_overlay, 0, 255).astype(np.uint8)
    
    # 保存整张ground truth叠加图
    full_gt_path = os.path.join(output_dir, f"{base_name}_full_gt.png")
    cv2.imwrite(full_gt_path, cv2.cvtColor(full_gt_overlay, cv2.COLOR_RGB2BGR))
    print(f"保存整张ground truth叠加图到: {full_gt_path}")
    # ======== 新增代码结束 ========
    
    # 保存预测掩码与真实掩码的对比图
    comparison_img = orig_img.copy()
    
    # 创建彩色掩码图层（整图对比）
    overlap = np.logical_and(gt_mask, pred_mask)
    only_gt = np.logical_and(gt_mask, np.logical_not(pred_mask))
    only_pred = np.logical_and(pred_mask, np.logical_not(gt_mask))
    
    # 修正颜色混合操作
    # 真实掩码但预测错误 - 绿色（半透明）
    green_color_arr = np.array([0, 200, 0], dtype=np.uint8)
    alpha = 0.7
    comparison_img[only_gt] = (comparison_img[only_gt].astype(np.float32) * (1 - alpha) + 
                           green_color_arr * alpha).astype(np.uint8)
    
    # 预测为前景但实际是背景 - 红色（半透明）
    red_color = np.array([200, 0, 0], dtype=np.uint8)
    comparison_img[only_pred] = (comparison_img[only_pred].astype(np.float32) * (1 - alpha) + 
                             red_color * alpha).astype(np.uint8)
    
    # 重叠区域（预测正确的前景）- 黄色（半透明）
    yellow_color = np.array([255, 255, 0], dtype=np.uint8)
    comparison_img[overlap] = (comparison_img[overlap].astype(np.float32) * (1 - alpha) + 
                           yellow_color * alpha).astype(np.uint8)
    
    # 保存对比图
    comparison_path = os.path.join(output_dir, f"{base_name}_full_comparison.png")
    cv2.imwrite(comparison_path, cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))
    print(f"保存整图对比结果到: {comparison_path}")
    
    return metrics

def calculate_average_metrics(metrics_list):
    """
    计算所有图像的平均评估指标
    """
    if not metrics_list:
        return {}
    
    # 初始化平均指标字典
    avg_metrics = {}
    
    # 获取所有指标名称（排除图像名称）
    metric_keys = [key for key in metrics_list[0].keys() if key != 'image_name']
    
    # 计算每个指标的平均值
    for key in metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            avg_metrics[key] = sum(values) / len(values)
        else:
            avg_metrics[key] = 0
    
    return avg_metrics

def save_metrics_to_file(avg_metrics, model_path, output_file):
    """
    将平均指标保存到文件
    """
    # 提取模型名称（使用路径的最后部分）
    model_name = os.path.basename(model_path)
    
    # 获取当前时间
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 准备要写入的行
    line = f"{timestamp} | {model_name} | "
    line += f"平均IoU: {avg_metrics.get('mean_iou', 0):.4f} | "
    line += f"前景IoU: {avg_metrics.get('iou_fg', 0):.4f} | "
    line += f"背景IoU: {avg_metrics.get('iou_bg', 0):.4f} | "
    line += f"平均Dice: {avg_metrics.get('mean_f1', 0):.4f} | "
    line += f"前景Dice: {avg_metrics.get('f1_fg', 0):.4f} | "
    line += f"准确率: {avg_metrics.get('accuracy', 0):.4f} | "
    line += f"平均精确率: {avg_metrics.get('mean_precision', 0):.4f} | "
    line += f"平均召回率: {avg_metrics.get('mean_recall', 0):.4f}"
    
    # 创建包含文件的目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 追加写入文件
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    
    print(f"\n已将评估结果保存到文件: {output_file}")
    return line

# ======== 使用示例 ========
if __name__ == "__main__":
    # 开始计时
    start_time = time.time()
    
    # 初始化模型
    unet = Unet()
    
    # 获取模型路径（用于记录）
    model_path = unet._defaults["model_path"]
    
    # === 在这里设置处理模式 ===
    process_mode = "folder"  # 可选: "single" 或 "folder"
    # =======================
    
    # 创建主输出目录
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_directory = f"/dataset/zhuluoji/unet/unet_double/output_new_test/{time_str}"
    os.makedirs(output_directory, exist_ok=True)
    
    # 指标记录文件路径
    metrics_file = "/dataset/zhuluoji/unet/unet_double/evaluation_metrics.txt"
    
    # 用于存储所有图像的评估指标
    all_metrics = []
    
    if process_mode == "single":
        # 处理单张图片
        image_path = "/dataset/zhuluoji/unet/unet_double/VOCdevkit3_copy/VOC2007/JPEGImages/6_right_front.jpg"
        metrics = process_single_image(image_path, output_directory, unet, all_metrics)
        
    elif process_mode == "folder":
        # 处理整个文件夹
        folder_path = "/dataset/zhuluoji/unet/unet_double/VOCdevkit3_copy/VOC2007/JPEGImages"
        
        # 获取所有JPG图像文件
        image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
        print(f"找到 {len(image_paths)} 张图片待处理")
        
        # 创建进度日志文件
        progress_log = os.path.join(output_directory, "progress.log")
        with open(progress_log, 'w', encoding='utf-8') as f:
            f.write(f"开始处理 {len(image_paths)} 张图片: {datetime.datetime.now()}\n")
        
        processed_count = 0
        success_count = 0
        
        for i, img_path in enumerate(image_paths):
            # 为每张图片创建单独的输出子目录
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_output_dir = os.path.join(output_directory, img_name)
            os.makedirs(img_output_dir, exist_ok=True)
            
            print(f"\n处理进度: {i+1}/{len(image_paths)} - {img_name}")
            try:
                metrics = process_single_image(img_path, img_output_dir, unet, all_metrics)
                processed_count += 1
                if metrics is not None:
                    success_count += 1
                    
                    # 将单张图像的指标保存到JSON文件
                    metrics_json_path = os.path.join(img_output_dir, "metrics.json")
                    with open(metrics_json_path, 'w', encoding='utf-8') as f:
                        json.dump(metrics, f, indent=4)
                
                # 更新进度日志
                with open(progress_log, 'a', encoding='utf-8') as f:
                    status = "成功" if metrics is not None else "失败"
                    f.write(f"[{i+1}/{len(image_paths)}] {img_name}: {status}\n")
                
            except Exception as e:
                print(f"处理图像 {img_name} 时出错: {str(e)}")
                # 记录错误到日志
                with open(progress_log, 'a', encoding='utf-8') as f:
                    f.write(f"[{i+1}/{len(image_paths)}] {img_name}: 错误 - {str(e)}\n")
                
                # 删除空目录（如果处理失败）
                if os.path.exists(img_output_dir) and not os.listdir(img_output_dir):
                    os.rmdir(img_output_dir)
        
        print(f"\n完成! 处理了 {processed_count}/{len(image_paths)} 张图片，成功: {success_count}")
    else:
        print(f"错误: 未知的处理模式 '{process_mode}'. 请使用 'single' 或 'folder'")
    
    # 计算平均指标
    if all_metrics:
        avg_metrics = calculate_average_metrics(all_metrics)
        
        # 打印平均指标
        print("\n========== 平均评估指标 ==========")
        print(f"平均IoU (前景): {avg_metrics['iou_fg']:.4f}")
        print(f"平均IoU (背景): {avg_metrics['iou_bg']:.4f}")
        print(f"平均IoU (所有类): {avg_metrics['mean_iou']:.4f}")
        print(f"平均Dice/F1 (前景): {avg_metrics['f1_fg']:.4f}")
        print(f"平均Dice/F1 (背景): {avg_metrics['f1_bg']:.4f}")
        print(f"平均Dice/F1 (所有类): {avg_metrics['mean_f1']:.4f}")
        print(f"平均准确率: {avg_metrics['accuracy']:.4f}")
        print(f"平均精确率 (所有类): {avg_metrics['mean_precision']:.4f}")
        print(f"平均召回率 (所有类): {avg_metrics['mean_recall']:.4f}")
        print("===================================")
        
        # 将平均指标保存到文件
        metrics_summary = save_metrics_to_file(avg_metrics, model_path, metrics_file)
        
                # 将平均指标保存到JSON文件
        avg_metrics_json_path = os.path.join(output_directory, "average_metrics.json")
        with open(avg_metrics_json_path, 'w', encoding='utf-8') as f:
            # 添加时间戳和处理信息
            result_data = {
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "model_path": model_path,
                "processed_images": len(all_metrics),
                "metrics": avg_metrics
            }
            json.dump(result_data, f, indent=4)
        
        # 创建详细的评估报告
        report_path = os.path.join(output_directory, "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("==========================================\n")
            f.write(f"分割评估报告 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("==========================================\n\n")
            f.write(f"模型路径: {model_path}\n")
            f.write(f"处理图像数量: {len(all_metrics)}\n\n")
            
            f.write("平均评估指标:\n")
            f.write(f"平均IoU (前景): {avg_metrics['iou_fg']:.4f}\n")
            f.write(f"平均IoU (背景): {avg_metrics['iou_bg']:.4f}\n")
            f.write(f"平均IoU (所有类): {avg_metrics['mean_iou']:.4f}\n")
            f.write(f"平均Dice/F1 (前景): {avg_metrics['f1_fg']:.4f}\n")
            f.write(f"平均Dice/F1 (背景): {avg_metrics['f1_bg']:.4f}\n")
            f.write(f"平均Dice/F1 (所有类): {avg_metrics['mean_f1']:.4f}\n")
            f.write(f"平均准确率: {avg_metrics['accuracy']:.4f}\n")
            f.write(f"平均精确率 (所有类): {avg_metrics['mean_precision']:.4f}\n")
            f.write(f"平均召回率 (所有类): {avg_metrics['mean_recall']:.4f}\n\n")
            
            # 添加每个图像的详细指标
            f.write("各图像指标详情:\n")
            f.write("------------------------------------------\n")
            for m in all_metrics:
                f.write(f"图像: {m['image_name']}\n")
                f.write(f"  IoU (前景): {m['iou_fg']:.4f}\n")
                f.write(f"  IoU (背景): {m['iou_bg']:.4f}\n")
                f.write(f"  平均IoU: {m['mean_iou']:.4f}\n")
                f.write(f"  Dice (前景): {m['f1_fg']:.4f}\n")
                f.write(f"  准确率: {m['accuracy']:.4f}\n")
                f.write("------------------------------------------\n")
        
        print(f"详细评估报告已保存到: {report_path}")
    
    # 计算并显示总运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_str = ""
    if hours > 0:
        time_str += f"{int(hours)}小时"
    if minutes > 0:
        time_str += f"{int(minutes)}分钟"
    time_str += f"{seconds:.2f}秒"
    
    print(f"\n总运行时间: {time_str}")
    
    # 将运行时间添加到进度日志
    if process_mode == "folder":
        with open(progress_log, 'a', encoding='utf-8') as f:
            f.write(f"\n处理完成，总运行时间: {time_str}\n")
            if all_metrics:
                f.write(f"平均IoU (所有类): {avg_metrics['mean_iou']:.4f}\n")
                f.write(f"平均Dice/F1 (所有类): {avg_metrics['mean_f1']:.4f}\n")
