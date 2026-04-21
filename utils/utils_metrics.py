import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes', num_classes)  
    #-----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    
    #------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    #------------------------------------------------#
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

    #------------------------------------------------#
    #   读取每一个（图片-标签）对
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))  

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )
    #------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    #------------------------------------------------#
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    #------------------------------------------------#
    #   逐类别输出一下mIoU值
    #------------------------------------------------#
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))  
    return np.array(hist, np.int32), IoUs, PA_Recall, Precision
def overlay_boundaries_on_image(orig_img, gt_mask, pred_mask):
    """
    在原图上叠加真实标签和预测结果的轮廓：
      - gt_mask: ground truth二值图（dtype可为np.uint8或其它，但值为0或1）
      - pred_mask: prediction二值图（值为0或1）
    返回叠加后的图像
    """
    # 为轮廓查找做好准备，转为uint8（注意：有些情况下可能需要乘以255）
    gt_mask_bin = (gt_mask.astype(np.uint8)) * 255
    pred_mask_bin = (pred_mask.astype(np.uint8)) * 255

    # 查找轮廓，注意cv2.findContours()在OpenCV不同版本返回值略有不同
    contours_gt, _ = cv2.findContours(gt_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pred, _ = cv2.findContours(pred_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 复制一份原图用于绘制结果（保证不会修改原图）
    overlay = orig_img.copy()

    # 绘制真实标签边界，red (BGR: (0,0,255))，线宽2
    cv2.drawContours(overlay, contours_gt, -1, (0, 0, 255), thickness=10)
    # 绘制预测结果边界，blue (BGR: (255,0,0))，线宽2
    cv2.drawContours(overlay, contours_pred, -1, (255, 0, 0), thickness=10)

    return overlay

def compute_mIoU_and_overlay(gt_dir, pred_dir, save_dir,
                             png_name_list, num_classes, name_classes=None):
    """
    参数说明：
      gt_dir: 真值（ground truth）分割图目录
      pred_dir: 预测结果分割图目录
      orig_dir: 原图目录（用于显示叠加轮廓效果）
      save_dir: 保存叠加效果图像的目录
      png_name_list: 图片名称列表（不带后缀）
      num_classes: 类别数（本例中为2：背景和目标）
      name_classes: 可选，每个类别的名称列表
    此函数既计算混淆矩阵和各项指标，也对每张图片生成轮廓叠加图像保存。
    """

    print('Num classes', num_classes)
    # 创建混淆矩阵
    hist = np.zeros((num_classes, num_classes))

    # 拼接各自的分割结果路径列表
    gt_imgs   = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for ind in range(len(gt_imgs)):
        # 读取预测结果和真值图像，并转换为numpy数组
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        # 如果两者尺寸不匹配，跳过该张图片
        if label.size != pred.size:
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                  label.size, pred.size, gt_imgs[ind], pred_imgs[ind]))
            continue

        # 累加混淆矩阵（这里假设已有fast_hist函数，用于计算单张图的混淆矩阵）
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        # 对每个10张图输出一次当前平均指标（需要已有per_class_iu, per_class_PA_Recall, per_Accuracy函数）
        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                  ind,
                  len(gt_imgs),
                  100 * np.nanmean(per_class_iu(hist)),
                  100 * np.nanmean(per_class_PA_Recall(hist)),
                  100 * per_Accuracy(hist)
            ))
            
        # -------------------------------
        # 叠加轮廓效果的保存部分
        # -------------------------------
        # 这里假设原图保存在orig_dir目录下，文件名与png_name_list中一致，后缀视情况而定（本例使用.jpg）
        orig_img_path = join(save_dir, png_name_list[ind] + "/rgb.png")
        if not os.path.exists(orig_img_path):
            print("原图不存在，无法生成叠加效果：", orig_img_path)
        else:
            orig_img = cv2.imread(orig_img_path)
            if orig_img is None:
                print("读取原图失败：", orig_img_path)
            else:
                # 生成叠加效果图
                overlay_img = overlay_boundaries_on_image(orig_img, label, pred)
                # 保存叠加效果图，可在文件名中加入后缀区分
                save_path = join(save_dir, png_name_list[ind] + "/mask_overlay.png")
                cv2.imwrite(save_path, overlay_img)
                print("保存叠加效果图：", save_path)
                
    # 计算总体的各项指标
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===> ' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) +
                  '; Recall- ' + str(round(PA_Recall[ind_class] * 100, 2)) +
                  '; Precision-' + str(round(Precision[ind_class] * 100, 2)))
            
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) +
          '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) +
          '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
    
    return np.array(hist, np.int32), IoUs, PA_Recall, Precision

def compute_all(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes:', num_classes)  
    # 创建一个全是0的混淆矩阵
    hist = np.zeros((num_classes, num_classes))
    
    # 构建验证集标签和预测结果路径列表
    gt_imgs   = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]  

    # 假设前景血管类别为1，背景为0
    foreground_class = 1
    background_class = 0
    
    # 存储每张图片的前景指标，用于后续分析
    per_image_metrics = []
    total_foreground_pixels = 0
    total_background_pixels = 0

    # 读取每个（图像-标签）对，计算累加混淆矩阵
    for ind in range(len(gt_imgs)): 
        # 读取预测结果和真值为numpy数组
        pred  = np.array(Image.open(pred_imgs[ind]))  
        label = np.array(Image.open(gt_imgs[ind]))  

        # 若二者尺寸不一致，则跳过该图
        if len(label.flatten()) != len(pred.flatten()):  
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue

        # 累加这张图片的混淆矩阵
        img_hist = fast_hist(label.flatten(), pred.flatten(), num_classes)
        hist += img_hist
        
        # 计算单张图片的前景指标
        TP = img_hist[foreground_class, foreground_class]
        FP = img_hist[background_class, foreground_class]
        FN = img_hist[foreground_class, background_class]
        TN = img_hist[background_class, background_class]
        
        # 计算前景像素比例
        foreground_pixels = np.sum(label == foreground_class)
        total_foreground_pixels += foreground_pixels
        total_background_pixels += np.sum(label == background_class)
        
        # 计算单张图片的前景指标
        if TP + FP > 0:
            img_precision = TP / (TP + FP)
        else:
            img_precision = 0
            
        if TP + FN > 0:
            img_recall = TP / (TP + FN)
        else:
            img_recall = 0
            
        if img_precision + img_recall > 0:
            img_f1 = 2 * img_precision * img_recall / (img_precision + img_recall)
        else:
            img_f1 = 0
            
        # 存储单张图片指标（按前景像素数加权）
        per_image_metrics.append({
            'precision': img_precision,
            'recall': img_recall,
            'f1': img_f1,
            'foreground_pixels': foreground_pixels
        })

    # 计算基于前景加权的指标
    foreground_weighted_precision = 0
    foreground_weighted_recall = 0
    foreground_weighted_f1 = 0
    total_weight = 0
    
    for img_metric in per_image_metrics:
        weight = img_metric['foreground_pixels']
        foreground_weighted_precision += img_metric['precision'] * weight
        foreground_weighted_recall += img_metric['recall'] * weight
        foreground_weighted_f1 += img_metric['f1'] * weight
        total_weight += weight
    
    if total_weight > 0:
        foreground_weighted_precision /= total_weight
        foreground_weighted_recall /= total_weight
        foreground_weighted_f1 /= total_weight
    
    # 计算各类指标
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    
    # 计算总体像素准确率
    overall_accuracy = np.sum(np.diag(hist)) / (np.sum(hist) + 1e-10)
    
    # 计算各类像素准确率
    class_acc = np.diag(hist) / (np.sum(hist, axis=1) + 1e-10)
    mean_accuracy = np.nanmean(class_acc)
    
    # 计算频权IoU
    freq = np.sum(hist, axis=1) / (np.sum(hist) + 1e-10)
    fwIoU = (freq[freq > 0] * IoUs[freq > 0]).sum()
    
    # 计算Dice系数 per 类
    dice = np.zeros(num_classes)
    for i in range(num_classes):
        denominator = np.sum(hist[i, :]) + np.sum(hist[:, i])
        if denominator == 0:
            dice[i] = float('nan')
        else:
            dice[i] = 2 * hist[i, i] / (denominator)
    mean_dice = np.nanmean(dice)
    
    # 计算各类别的Recall（召回率）
    Recall = np.zeros(num_classes)
    for i in range(num_classes):
        TP = hist[i, i]
        FN = np.sum(hist[i, :]) - TP
        if TP + FN == 0:
            Recall[i] = float('nan')
        else:
            Recall[i] = TP / (TP + FN)
    mean_recall = np.nanmean(Recall)
    
    # 计算F1-score per 类
    F1_score = np.zeros(num_classes)
    for i in range(num_classes):
        if Precision[i] + Recall[i] == 0:
            F1_score[i] = float('nan')
        else:
            F1_score[i] = 2 * Precision[i] * Recall[i] / (Precision[i] + Recall[i])
    mean_f1 = np.nanmean(F1_score)
    
    # 计算特异性（Specificity）per 类
    Specificity = np.zeros(num_classes)
    for i in range(num_classes):
        TN = np.sum(hist) - np.sum(hist[i, :]) - np.sum(hist[:, i]) + hist[i, i]
        FP = np.sum(hist[:, i]) - hist[i, i]
        if TN + FP == 0:
            Specificity[i] = float('nan')
        else:
            Specificity[i] = TN / (TN + FP)
    mean_specificity = np.nanmean(Specificity)
    
    # 计算Kappa系数
    total = np.sum(hist)
    po = overall_accuracy
    pe = 0.0
    for i in range(num_classes):
        pe += (np.sum(hist[i, :]) * np.sum(hist[:, i]))
    pe = pe / (total * total) if total > 0 else 0
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
    
    # ========== 新增针对不平衡数据的指标 ==========
    # 1. 计算G-Mean（几何平均）
    g_mean = np.sqrt(Recall[foreground_class] * Specificity[foreground_class])
    
    # 2. 计算前景平衡准确率
    foreground_balanced_accuracy = (Recall[foreground_class] + Specificity[foreground_class]) / 2
    
    # 3. 计算前景与背景的像素比例
    foreground_ratio = total_foreground_pixels / (total_foreground_pixels + total_background_pixels)
    
    # 前景类别指标
    foreground_iou = IoUs[foreground_class] if not np.isnan(IoUs[foreground_class]) else 0
    foreground_precision = Precision[foreground_class] if not np.isnan(Precision[foreground_class]) else 0
    foreground_recall = Recall[foreground_class] if not np.isnan(Recall[foreground_class]) else 0
    foreground_f1 = F1_score[foreground_class] if not np.isnan(F1_score[foreground_class]) else 0
    foreground_dice = dice[foreground_class] if not np.isnan(dice[foreground_class]) else 0
    foreground_specificity = Specificity[foreground_class] if not np.isnan(Specificity[foreground_class]) else 0
    
    # 输出指标（重点突出前景相关指标）
    print('===> Foreground Ratio: {:0.4f}'.format(foreground_ratio))
    print('===> Overall Accuracy: {:0.2f}%; Kappa: {:0.2f}%'.format(
        round(overall_accuracy * 100, 2), round(kappa * 100, 2)))
    print('===> Foreground Metrics:')
    print('     IoU: {:0.2f}%; Precision: {:0.2f}%; Recall: {:0.2f}%'.format(
        round(foreground_iou * 100, 2), round(foreground_precision * 100, 2), 
        round(foreground_recall * 100, 2)))
    print('     F1: {:0.2f}%; Dice: {:0.2f}%; Specificity: {:0.2f}%'.format(
        round(foreground_f1 * 100, 2), round(foreground_dice * 100, 2),
        round(foreground_specificity * 100, 2)))
    print('     G-Mean: {:0.2f}%; Balanced Accuracy: {:0.2f}%'.format(
        round(g_mean * 100, 2), round(foreground_balanced_accuracy * 100, 2)))
    print('===> Foreground-Weighted Metrics (per-image):')
    print('     Precision: {:0.2f}%; Recall: {:0.2f}%; F1: {:0.2f}%'.format(
        round(foreground_weighted_precision * 100, 2), 
        round(foreground_weighted_recall * 100, 2),
        round(foreground_weighted_f1 * 100, 2)))
    # 1. 添加G-Mean指标
    g_mean = np.sqrt(Recall[foreground_class] * Specificity[foreground_class])

    # 2. 添加平衡准确率
    foreground_balanced_accuracy = (Recall[foreground_class] + Specificity[foreground_class]) / 2

    # 3. 计算前景像素占比
    total_pixels = np.sum(hist)
    foreground_ratio = np.sum(hist[foreground_class, :]) / total_pixels

    
    
    # 将所有指标以字典形式返回
    metrics = {
        'confusion_matrix': np.array(hist, np.int32),
        
        # 整体指标
        'overall_accuracy': overall_accuracy,
        'mean_accuracy': mean_accuracy,
        'frequency_weighted_IoU': fwIoU,
        'kappa': kappa,
        
        # 平均指标
        'mean_IoU': np.nanmean(IoUs),
        'mean_Precision': np.nanmean(Precision),
        'mean_Recall': mean_recall,
        'mean_F1_score': mean_f1,
        'mean_Dice': mean_dice,
        'mean_Specificity': mean_specificity,
        
        # 前景类别指标
        'foreground_IoU': foreground_iou,
        'foreground_Precision': foreground_precision,
        'foreground_Recall': foreground_recall,
        'foreground_F1_score': foreground_f1,
        'foreground_Dice': foreground_dice,
        'foreground_Specificity': foreground_specificity,
        
        # 新增不平衡数据专用指标
        'foreground_G_mean': g_mean,
        'foreground_balanced_accuracy': foreground_balanced_accuracy,
        'foreground_ratio': foreground_ratio,
        
        # 前景加权指标
        'foreground_weighted_precision': foreground_weighted_precision,
        'foreground_weighted_recall': foreground_weighted_recall,
        'foreground_weighted_f1': foreground_weighted_f1,
        
        # 每类指标数组
        'per_class_IoU': IoUs,
        'per_class_Precision': Precision,
        'per_class_Recall': Recall,
        'per_class_F1_score': F1_score,
        'per_class_Dice': dice
    }
    # 在metrics字典中添加：
    metrics.update({
        'foreground_G_mean': g_mean,
        'foreground_balanced_accuracy': foreground_balanced_accuracy, 
        'foreground_ratio': foreground_ratio,
    })
    return metrics


def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            