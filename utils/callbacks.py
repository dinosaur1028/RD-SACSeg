import os

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
import csv

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image,resize_gray_image
from .utils_metrics import compute_mIoU,compute_mIoU_and_overlay,compute_all
import datetime


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir        = log_dir
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss = None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
            
        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
            
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")
now = datetime.datetime.now()

# 格式化日期时间字符串，例如："2023-10-20 14:30:15"
t = now.strftime("%Y-%m-%d %H:%M:%S")
class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=f".temp_miou_out{t}", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious      = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image, dex, lab, name=None,log_dir=None,epoch=None):
        #---------------------------------------------------------#
        #   转化为RGB图像，防止灰度图在预测时报错（代码仅支持RGB图像）
        #---------------------------------------------------------#
        image       = cvtColor(image)
        original_h  = np.array(image).shape[0]
        original_w  = np.array(image).shape[1]
        
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   这里调用自定义的 resize_image 函数，返回调整后的图像、宽w和高h
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上 batch_size 维度，转换为 (1, 3, H, W)
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        #---------------------------------------------------------#
        #   处理dex图像：resize、转换为numpy、归一化，并扩展为 (1,1,H,W)
        #---------------------------------------------------------#
        dex, _, _  = resize_gray_image(dex, (self.input_shape[1], self.input_shape[0]))
        dex_array = np.array(dex, np.float32) / 255.0
        dex_array = np.expand_dims(np.expand_dims(dex_array, axis=0), axis=0)
        
        #---------------------------------------------------------#
        #   处理lab图像：resize、转换为numpy、归一化，并扩展为 (1,1,H,W)
        #---------------------------------------------------------#
        lab, _, _  = resize_gray_image(lab, (self.input_shape[1], self.input_shape[0]))
        lab_array = np.array(lab, np.float32) / 255.0
        lab_array = np.expand_dims(np.expand_dims(lab_array, axis=0), axis=0)

        with torch.no_grad():
            # 转为tensor
            images = torch.from_numpy(image_data)
            dex_tensor = torch.from_numpy(dex_array)
            lab_tensor = torch.from_numpy(lab_array)
            
            # 沿着通道维度拼接：RGB (3通道) + dex (1通道) + lab (1通道) 
            images = torch.cat([images, dex_tensor, lab_tensor], dim=1)
            
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images,name,log_dir,epoch)[0]
            #---------------------------------------------------#
            #   取出每个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            #--------------------------------------#
            #   截取掉灰条部分
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   resize 回原始尺寸
            #---------------------------------------------------#
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每个像素点的种类索引
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
        image_out = Image.fromarray(np.uint8(pr))
        if epoch%100 == 0:
            pr_scaled = np.clip(pr * 255, 0, 255)
            image_scaled_out = Image.fromarray(pr_scaled.astype(np.uint8))
            save_dir = os.path.join(f"{log_dir}/{epoch}", name)
            image_scaled_out.save(os.path.join(save_dir, "result.png"))
        return image_out
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass_Aug/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#
                image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages_Aug/"+image_id+".jpg")
                image       = Image.open(image_path)

                dex_path = os.path.join(self.dataset_path, "VOC2007/Dexined_Aug", image_id + ".png")
                dex   = Image.open(dex_path).convert("L")  # 保证为灰度图

                lab_path = os.path.join(self.dataset_path, "VOC2007/LAB_Aug", image_id + ".png")
                lab   = Image.open(lab_path).convert("L")  # 保证为灰度图
                
                

                
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                image       = self.get_miou_png(image,dex,lab,image_id,self.log_dir,epoch)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate miou.")
            # _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            metrics = compute_all(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = metrics['mean_IoU']
            # temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")

            output_csv = os.path.join(self.log_dir, "metrics.csv")

            # 判断文件是否存在且非空，如果不存在或为空，则写入表头
            write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0

            with open(output_csv, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                
                # 如果需要写入表头，则写入 "Epoch" + 各指标名称（保证顺序一致）
                if write_header:
                    header = ["Epoch"] + list(metrics.keys())
                    writer.writerow(header)
                
                # 构造本次写入的一行数据，第一列为epoch，随后依次是指标对应值
                row = [epoch]
                for key in metrics.keys():
                    value = metrics[key]
                    # 如果指标为 numpy 数组，则转换为列表后使用逗号分隔转换成字符串
                    if isinstance(value, np.ndarray):
                        value_str = ", ".join(map(str, value.tolist()))
                    # 如果指标为列表，也转换为以逗号分隔的字符串
                    elif isinstance(value, list):
                        value_str = ", ".join(map(str, value))
                    else:
                        value_str = str(value)
                    row.append(value_str)
                
                writer.writerow(row)
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
