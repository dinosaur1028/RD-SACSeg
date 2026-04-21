import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
import os
from nets.unet_final_0 import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和num_classes都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的model_path和num_classes数的修改
#--------------------------------------------#
class Unet(object):
    _defaults = {
        #ours
        "model_path"    : '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-11-06 20:11:08/best_epoch_weights.pth',
        
        #-------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        #-------------------------------------------------------------------#（1）
        # "model_path"    : '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep005-loss0.449-val_loss0.435.pth',
        # "model_path"    : '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep050-loss0.388-val_loss0.381.pth',
        # "model_path"    : '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep100-loss0.372-val_loss0.363.pth',
        # "model_path"    : '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep150-loss0.344-val_loss0.350.pth',
        # "model_path"    : '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep200-loss0.322-val_loss0.342.pth',
        # "model_path"    : '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep015-loss0.416-val_loss0.407.pth',
        # "model_path"    : '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep025-loss0.404-val_loss0.395.pth',
        #-------------------------------------------------------------------#（2）
        # "model_path"    : '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/best_epoch_weights.pth',
       
        #--------------------------------#
        #   所需要区分的类的个数+1
        #--------------------------------#
        "num_classes"   : 2,
        #--------------------------------#
        #   所使用的的主干网络：vgg、resnet50   
        #--------------------------------#
        "backbone"      : "vgg",
        #--------------------------------#
        #   输入图片的大小
        #--------------------------------#
        # "input_shape"   : [448, 768],
        "input_shape"   : [512, 512],
        #-------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        #-------------------------------------------------#
        "mix_type"      : 1,
        #--------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #--------------------------------#
        "cuda"          : True,
    }

    #---------------------------------------------------#
    #   初始化UNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   获得模型
        #---------------------------------------------------#
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = unet(num_classes = self.num_classes, backbone=self.backbone)

        device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                # self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, dex, lab, count=False, name_classes=None):
        """
        image : 需要检测的原始图片 (RGB)
        dex   : 对应的 dex 灰度图
        lab   : 对应的 lab 灰度图
        count : 是否统计各类别像素像素数量
        name_classes : 类别名称列表
        """
        # 将原始图片统一转换为RGB
        image = cvtColor(image)
        # 保存一份原图（用于后续绘制轮廓）
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # 对 RGB 图片进行 resize（不失真），获得调整后的图片及新宽高
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # 将 RGB 图片处理为 numpy 数组，并添加 batch_size 维度，(1, 3, H, W)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        # 处理 dex 图片：resize、转换为 numpy 数组、归一化，并扩展为 (1, 1, H, W)
        dex_resized, _, _ = resize_gray_image(dex, (self.input_shape[1], self.input_shape[0]))
        dex_array = np.array(dex_resized, np.float32) / 255.0
        dex_array = np.expand_dims(np.expand_dims(dex_array, axis=0), axis=0)

        # 处理 lab 图片：resize、转换为 numpy 数组、归一化，并扩展为 (1, 1, H, W)
        lab_resized, _, _ = resize_gray_image(lab, (self.input_shape[1], self.input_shape[0]))
        lab_array = np.array(lab_resized, np.float32) / 255.0
        lab_array = np.expand_dims(np.expand_dims(lab_array, axis=0), axis=0)

        with torch.no_grad():
            # 转为 tensor
            rgb_tensor = torch.from_numpy(image_data)  # (1,3,H,W)
            dex_tensor = torch.from_numpy(dex_array)     # (1,1,H,W)
            lab_tensor = torch.from_numpy(lab_array)     # (1,1,H,W)

            # 拼接 3 通道 RGB、1 通道 dex、1 通道 lab，构成 5 通道输入
            images = torch.cat([rgb_tensor, dex_tensor, lab_tensor], dim=1)  # (1,5,H,W)

            if self.cuda:
                images = images.cuda()

            # 网络预测，返回的 pr 为 (C, H, W)，取第一张图
            pr = self.net(images)[0]

            # 对预测结果进行 softmax，并变换为 (H, W, C) 后计算 argmax
            pr = torch.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # 去除 resize 时可能添加的灰条
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # 将结果 resize 回原图尺寸
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # 得到每个像素对应的类别索引
            pr = pr.argmax(axis=-1)

        # 统计类别像素数量（如果需要）
        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        # 根据 mix_type 选择分割结果的可视化方式
        if self.mix_type == 0:
            # 生成彩色分割图，然后与原图混合
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                                [orininal_h, orininal_w, -1])
            image_seg = Image.fromarray(np.uint8(seg_img))
            image_out = Image.blend(old_img, image_seg, 0.7)
        elif self.mix_type == 1:
            # 直接输出彩色分割图
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                                [orininal_h, orininal_w, -1])
            image_out = Image.fromarray(np.uint8(seg_img))
        elif self.mix_type == 2:
            # 扣除背景，仅保留目标
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image_out = Image.fromarray(np.uint8(seg_img))
        else:
            image_out = old_img

        # 额外输出一张图：在原图上绘制分割掩码的轮廓
        # 注意：这里我们认为 pr 中非 0 的区域为目标区域，可以根据需要修改
        image_np = np.array(old_img)  # PIL -> numpy，RGB格式
        # 构造二值掩码：目标区域为 255，背景为 0
        mask = np.where(pr != 0, 255, 0).astype(np.uint8)
        # 查找轮廓 (opencv 需要单通道二值图)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 在原图上绘制轮廓，注意 cv2.drawContours 要使用 BGR 颜色，这里我们选用蓝色： (255, 0, 0)
        image_with_contour = image_np.copy()
        cv2.drawContours(image_with_contour, contours, -1, (255, 0, 0), thickness=2)
        # 若希望输出的图像为 PIL Image 格式，则需要转换颜色空间
        image2 = Image.fromarray(image_with_contour)

        return image_out, image2
    

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------#
                #   图片传入网络进行预测
                #---------------------------------------------------#
                pr = self.net(images)[0]
                #---------------------------------------------------#
                #   取出每一个像素点的种类
                #---------------------------------------------------#
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                #--------------------------------------#
                #   将灰条部分截取掉
                #--------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image

def resize_gray_image(image, size):
    iw, ih = image.size
    w, h   = size

    scale = min(w/iw, h/ih)
    nw    = int(iw * scale)
    nh    = int(ih * scale)

    image_resized = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('L', size, 128)  # 'L'模式灰度图，初始填充值为128
    new_image.paste(image_resized, ((w - nw) // 2, (h - nh) // 2))
    return new_image, nw, nh