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

def process_single_image(image_path, output_dir, unet):
    """
    处理单张图像的函数
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
        return
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    
    # 读取真实掩码并二值化
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"错误: 无法读取mask {gt_mask_path}")
        return
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
            return
    
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

# ======== 使用示例 ========
if __name__ == "__main__":
    unet = Unet()
    
    # === 在这里设置处理模式 ===
    process_mode = "folder"  # 可选: "single" 或 "folder"
    # =======================
    
    # 创建主输出目录
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_directory = f"/dataset/zhuluoji/unet/unet_double/output_new_test/{time_str}"
    os.makedirs(output_directory, exist_ok=True)
    
    if process_mode == "single":
        # 处理单张图片
        image_path = "/dataset/zhuluoji/unet/unet_double/VOCdevkit3_copy/VOC2007/JPEGImages/6_right_front.jpg"
        process_single_image(image_path, output_directory, unet)
        
    elif process_mode == "folder":
        # 处理整个文件夹
        folder_path = "/dataset/zhuluoji/unet/unet_double/VOCdevkit3_copy/VOC2007/JPEGImages"
        
        # 获取所有JPG图像文件
        image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
        print(f"找到 {len(image_paths)} 张图片待处理")
        
        for i, img_path in enumerate(image_paths):
            # 为每张图片创建单独的输出子目录
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_output_dir = os.path.join(output_directory, img_name)
            os.makedirs(img_output_dir, exist_ok=True)
            
            print(f"\n处理进度: {i+1}/{len(image_paths)} - {img_name}")
            try:
                process_single_image(img_path, img_output_dir, unet)
            except Exception as e:
                print(f"处理图像 {img_name} 时出错: {str(e)}")
                # 删除空目录（如果处理失败）
                if os.path.exists(img_output_dir) and not os.listdir(img_output_dir):
                    os.rmdir(img_output_dir)
        
        print(f"\n完成! 处理了 {len(image_paths)} 张图片")
    else:
        print(f"错误: 未知的处理模式 '{process_mode}'. 请使用 'single' 或 'folder'")
