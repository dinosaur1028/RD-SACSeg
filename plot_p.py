import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class ParameterLogger:
    def __init__(self, log_dir, model):
        """
        初始化参数记录器
        
        Args:
            log_dir: 日志保存目录
            model: 模型实例
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.model = model
        self.writer = SummaryWriter(log_dir)
        
        # 初始化参数历史记录字典（区分 region 和 detail 两组带通参数）
        self.param_history = {
            # Region（learnable_bandpass_region 的参数）
            'region_lower_bound': [],
            'region_upper_bound': [],
            'region_tau': [],
            # Detail（learnable_bandpass_detail 的参数）
            'detail_lower_bound': [],
            'detail_upper_bound': [],
            'detail_tau': [],
            # 融合模块参数（可选记录，用于 TensorBoard 观察）
            'alpha': [],
            'beta': []
        }
        
    def log_parameters(self, epoch):
        """
        记录当前epoch的参数值
        
        Args:
            epoch: 当前训练的epoch
        """
        # 获取区域带通参数（learnable_bandpass_region）
        region_lower_bound = self.model.learnable_bandpass_region.lower_bound.item()
        region_upper_bound = self.model.learnable_bandpass_region.upper_bound.item()
        region_tau = self.model.learnable_bandpass_region.tau.item()
        
        # 获取细节带通参数（learnable_bandpass_detail）
        detail_lower_bound = self.model.learnable_bandpass_detail.lower_bound.item()
        detail_upper_bound = self.model.learnable_bandpass_detail.upper_bound.item()
        detail_tau = self.model.learnable_bandpass_detail.tau.item()
        
        # 获取特征融合模块的参数（若存在）
        alpha = self.model.fusion_module.alpha.item()
        beta = self.model.fusion_module.beta.item()
        
        # 添加到历史记录
        self.param_history['region_lower_bound'].append(region_lower_bound)
        self.param_history['region_upper_bound'].append(region_upper_bound)
        self.param_history['region_tau'].append(region_tau)
        
        self.param_history['detail_lower_bound'].append(detail_lower_bound)
        self.param_history['detail_upper_bound'].append(detail_upper_bound)
        self.param_history['detail_tau'].append(detail_tau)
        
        self.param_history['alpha'].append(alpha)
        self.param_history['beta'].append(beta)
        
        # 写入TensorBoard（区域 + 细节两组参数）
        self.writer.add_scalar('Parameters/region_lower_bound', region_lower_bound, epoch)
        self.writer.add_scalar('Parameters/region_upper_bound', region_upper_bound, epoch)
        self.writer.add_scalar('Parameters/region_tau', region_tau, epoch)
        
        self.writer.add_scalar('Parameters/detail_lower_bound', detail_lower_bound, epoch)
        self.writer.add_scalar('Parameters/detail_upper_bound', detail_upper_bound, epoch)
        self.writer.add_scalar('Parameters/detail_tau', detail_tau, epoch)
        
        self.writer.add_scalar('Parameters/alpha', alpha, epoch)
        self.writer.add_scalar('Parameters/beta', beta, epoch)
        
        # 每隔一定epoch保存参数变化曲线图
        if epoch % 10 == 0:
            self.plot_parameter_curves(epoch)
    
    def plot_parameter_curves(self, epoch):
        """
        绘制参数变化曲线并保存
        将分别绘制两张图：
        - 图1：Region 的 region_lower_bound、region_upper_bound、region_tau
        - 图2：Detail 的 detail_lower_bound、detail_upper_bound、detail_tau
        
        Args:
            epoch: 当前训练的epoch
        """
        # 图1：Region 参数曲线
        epochs_region = list(range(1, len(self.param_history['region_lower_bound']) + 1))
        
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_region, self.param_history['region_lower_bound'], 'b-', label='Region Lower Bound')
        plt.plot(epochs_region, self.param_history['region_upper_bound'], 'r-', label='Region Upper Bound')
        plt.plot(epochs_region, self.param_history['region_tau'], 'g-', label='Region Tau')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Learnable Region BandPass Parameters')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'parameter_curves_region_epoch_{epoch}.png'))
        plt.close()
        
        # 图2：Detail 参数曲线
        epochs_detail = list(range(1, len(self.param_history['detail_lower_bound']) + 1))
        
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_detail, self.param_history['detail_lower_bound'], 'b-', label='Detail Lower Bound')
        plt.plot(epochs_detail, self.param_history['detail_upper_bound'], 'r-', label='Detail Upper Bound')
        plt.plot(epochs_detail, self.param_history['detail_tau'], 'g-', label='Detail Tau')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Learnable Detail BandPass Parameters')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'parameter_curves_detail_epoch_{epoch}.png'))
        plt.close()
    
    def save_final_curves(self):
        """
        训练结束时保存最终的参数变化曲线
        """
        # 最终绘图（根据两张图单独保存）
        # 这里直接调用两张图的保存逻辑，传入一个特殊标记即可
        self.plot_parameter_curves('final')
        
        # 额外保存参数数据为CSV文件以便后续分析
        import pandas as pd
        df = pd.DataFrame(self.param_history)
        df.index = list(range(1, len(df) + 1))  # 设置epoch作为索引
        df.index.name = 'epoch'
        df.to_csv(os.path.join(self.log_dir, 'parameter_history.csv'))
        
        # 关闭TensorBoard写入器
        self.writer.close()
