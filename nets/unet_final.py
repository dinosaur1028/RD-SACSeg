#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import resnet50
from nets.vgg import VGG16

# 上采样模块（保持原来 unetUp 结构不变）
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up    = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu  = nn.ReLU(inplace=True)
    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], dim=1)
        outputs = self.relu(self.conv1(outputs))
        outputs = self.relu(self.conv2(outputs))
        return outputs

# Squeeze-and-Excitation特征融合模块
class HandFeatureFusion(nn.Module):
    def __init__(self, in_channels, mid_channels):
        """
        in_channels: 分别 detail 和 region 分支的输出通道数（假设二者一致）
        mid_channels: 融合后特征的通道数
        """
        super(HandFeatureFusion, self).__init__()
        # 级联后通道数为 2*in_channels，之后压缩到 mid_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # Squeeze-and-Excitation模块，全局上下文提取和通道加权
        self.fc = nn.Sequential(
            nn.Linear(mid_channels, mid_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels // 16, mid_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, detail_feat, region_feat):
        # detail_feat 和 region_feat 形状均为 (B, C, H, W)
        x = torch.cat([detail_feat, region_feat], dim=1)  # (B, 2*C, H, W)
        out = self.conv(x)                                # (B, mid_channels, H, W)
        # 全局平均池化获得通道描述子
        b, c, h, w = out.size()
        y = F.adaptive_avg_pool2d(out, 1).view(b, c)        # (B, c)
        y = self.fc(y).view(b, c, 1, 1)                     # (B, c, 1, 1)
        out = out * y                                     # 通道加权
        return out

# ----------------------
# LearnableBandPass 模块：利用两个 sigmoid 自动学习 detail_map 的带通区域
class LearnableBandPass(nn.Module):
    def __init__(self, init_lower=0.1, init_upper=0.4, tau=0.05):
        """
        init_lower: 初始化的 lower bound（归一化后值）
        init_upper: 初始化的 upper bound（归一化后值）
        tau: 控制过渡平滑程度（可固定，也可设为可学习参数）
        """
        super(LearnableBandPass, self).__init__()
        # 定义可学习参数
        self.lower_bound = nn.Parameter(torch.tensor(init_lower, dtype=torch.float32))
        self.upper_bound = nn.Parameter(torch.tensor(init_upper, dtype=torch.float32))
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))

    def forward(self, detail_map):
        """
        detail_map: [B,1,H,W]，假设已归一化到 [0,1]
        输出：权重图 [B,1,H,W]，使得 detail_map 中介于 lower_bound 与 upper_bound
             区间的区域权重较高，其他区域权重较低。
        """
        w1 = torch.sigmoid((detail_map - self.lower_bound) / self.tau)
        w2 = torch.sigmoid((detail_map - self.upper_bound) / self.tau)
        weight = w1 - w2
        return weight
# ----------------------

# 带有手工特征信息的 Unet
class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        self.backbone = backbone
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained, in_channels=3)
            # 假定 feat1 的输出通道数依据具体 VGG16 实现
            in_filters  = [192, 384, 768, 1024]
            base_feat_ch = 64  # 根据实际情况调整
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters  = [192, 512, 1024, 3072]
            base_feat_ch = 64  # 根据具体实现调整
        else:
            raise ValueError(f'Unsupported backbone - {backbone}')
        
        # 针对手工特征设计简单卷积模块，stride 决定下采样比例
        self.detail_conv = nn.Sequential(
            nn.Conv2d(1, base_feat_ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # 新增：LearnableBandPass 用于 detail_map 的带通加权
        self.learnable_bandpass = LearnableBandPass(init_lower=0.1, init_upper=0.4, tau=0.05)
        
        self.region_conv = nn.Sequential(
            nn.Conv2d(1, base_feat_ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(base_feat_ch, base_feat_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder部分
        out_filters = [64, 128, 256, 512]
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        
        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.up_conv = None
        
        self.final = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
        self.fusion_module = HandFeatureFusion(in_channels=base_feat_ch, mid_channels=base_feat_ch)
    
    def forward(self, inputs, img_name=None,log_dir=None,epoch=None):
        def save_feat_attention(feature_tensor, name):
                # 求均值作为注意力图，并调整到原图尺寸
                feat_att = torch.mean(feature_tensor, dim=1, keepdim=True)
                feat_att = F.interpolate(feat_att, size=rgb_np.shape[:2], mode='bilinear', align_corners=True)
                feat_att_np = feat_att[0,0].detach().cpu().numpy()
                # 归一化处理
                if feat_att_np.max() - feat_att_np.min() > 1e-8:
                    feat_att_norm = (feat_att_np - feat_att_np.min()) / (feat_att_np.max() - feat_att_np.min())
                else:
                    feat_att_norm = feat_att_np - feat_att_np.min()
                feat_att_norm = np.uint8(feat_att_norm * 255)
                feat_heat = cv2.applyColorMap(feat_att_norm, cv2.COLORMAP_JET)
                feat_overlay = cv2.addWeighted(rgb_bgr, 0.5, feat_heat, 0.5, 0)
                cv2.imwrite(os.path.join(save_dir, f"{name}.png"), feat_overlay)
        # inputs: [B,5,W,H]
        img = inputs[:, :3, :, :]         # RGB 图像输入
        detail_map = inputs[:, 3:4, :, :]   # 注重细节的手工特征
        region_map = inputs[:, 4:5, :, :]   # 注重区域的手工特征
        
        # 主干提取图像特征
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(img)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(img)
        
         # 细节分支处理：先用 detail_conv 获得基础特征，再利用 learnable_bandpass 进行权重调制
        detail_feat = self.detail_conv(detail_map)
        weight = self.learnable_bandpass(detail_map)
        if weight.shape[2:] != detail_feat.shape[2:]:
            weight = F.interpolate(weight, size=detail_feat.shape[2:], mode='bilinear', align_corners=True)
        detail_feat = detail_feat * weight  # 带通加权后的细节特征

        # 区域分支处理
        region_feat = self.region_conv(region_map)
        # 融合两个手工特征分支
        # fused_hand_feature = self.fusion_module(detail_feat, region_feat)
        fused_hand_feature = detail_feat
        
        # 当 fused_hand_feature 与 feat1 空间尺寸不一致时进行上采样匹配
        if fused_hand_feature.shape[2:] != feat1.shape[2:]:
            fused_hand_feature = F.interpolate(fused_hand_feature, size=feat1.shape[2:], mode='bilinear', align_corners=True)
        
        # 融合手工特征与第一层特征，并通过 fuse_conv 处理
        fused_feat1 = self.fuse_conv(feat1 + fused_hand_feature)
        
        # Decoder 上采样过程
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(fused_feat1, up2)
        up1_o = self.up_concat1(feat1, up2)
        
        if self.up_conv is not None:
            up1 = self.up_conv(up1)
        
        final = self.final(up1)
        # detail_att = F.interpolate(detail_feat, size=final.shape[2:], mode='bilinear', align_corners=True)
        # # 如果 detail_att 没有归一化，则归一化到 [0,1]（可根据具体情况调整）
        # dmin, dmax = detail_att.min(), detail_att.max()
        # if dmax - dmin > 1e-8:
        #     detail_att = (detail_att - dmin) / (dmax - dmin)
        # else:
        #     detail_att = detail_att - dmin
        # # 将 final 与注意力权重相乘，完成调制（element-wise乘法）
        # final = final * detail_att

        # 如果传入了 img_name 参数，则将中间结果以图片形式存储
        if img_name is not None and epoch%100==0:
            # 创建保存结果的文件夹：mid_result/img_name/
            os.makedirs(f"{log_dir}/{epoch}", exist_ok=True)
            os.makedirs(f"{log_dir}/{epoch}/mid_result", exist_ok=True)
            save_dir = os.path.join(f"{log_dir}/{epoch}/mid_result", img_name)
            os.makedirs(save_dir, exist_ok=True)

            # 默认 batch_size==1，且输入图像的像素范围为 [0,1]，根据需要调整
            # 处理 RGB 图像
            rgb_tensor = img[0].detach().cpu()  # (3,H,W)
            rgb_np = rgb_tensor.permute(1,2,0).numpy()  # (H,W,3)
            rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, "rgb.png"), rgb_bgr)

            # 处理 detail_map（灰度图）
            detail_tensor = detail_map[0,0].detach().cpu().numpy()
            detail_np = np.clip(detail_tensor * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, "detail_map.png"), detail_np)
            
            # 处理 region_map（灰度图）
            region_tensor = region_map[0,0].detach().cpu().numpy()
            region_np = np.clip(region_tensor * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, "region_map.png"), region_np)

            # 生成 detail_feat 注意力图：取均值、插值、归一化、伪彩色，再叠加原图
            save_feat_attention(detail_feat, 'detail_feat')

            # 生成 region_feat 注意力图，方法同上
            save_feat_attention(region_feat, 'region_feat')


            # ------------------------------
            # 新增：保存 fused_hand_feature 的注意力图
            save_feat_attention(fused_hand_feature, 'fused_hand_feature')

            # ------------------------------
            # 保存 backbone 提取的特征图 feat1~feat5
            feats = [feat1, feat2, feat3, feat4, feat5]
            feat_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]
            for feat, name in zip(feats, feat_names):
                save_feat_attention(feat, name)

            # 保存解码器过程中的特征图：up4, up3, up2, up1
            decoder_feats = [up4, up3, up2, up1,up1_o]
            decoder_names = ["up4", "up3", "up2", "up1","up1_o"]
            for feat, name in zip(decoder_feats, decoder_names):
                save_feat_attention(feat, name)
            # ------------------------------
            # 保存 final 在原图上的叠加结果
            # 这里取 final 的各通道均值作为注意力信息（也可根据需要调整为 softmax 或取最大响应）
            save_feat_attention(final, 'final')

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True