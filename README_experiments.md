# 实验代码说明文档
> 针对审稿意见的补充实验，共12个实验文件

---

## 文件清单

| 文件 | 对应实验 | 审稿人意见 |
|------|---------|-----------|
| `exp1_kfold_cv.py` | 5折交叉验证 + 5种子重复 | 审稿人1-意见3, 审稿人2-意见1 |
| `exp2_wilcoxon_test.py` | 配对Wilcoxon检验 | 审稿人2-意见4 |
| `exp3_efficiency.py` | FLOPs/参数量/推理速度 | 审稿人1-意见10 |
| `exp4_roc_auc.py` | 像素级ROC曲线+AUC | 审稿人1-意见11 |
| `exp5_fisher_clinical.py` | Fisher精确检验（临床） | 审稿人1-意见7, 审稿人2-意见5 |
| `exp6_new_models.py` | 新对比模型(MedSAM/VMamba/YOLOv8) | 审稿人1-意见4 |
| `exp7_lbf_ablation.py` | LBF超参数消融(γ值+初始化策略) | 审稿人1-意见12 |
| `exp8_lbf_convergence.py` | LBF梯度推导+α/β收敛曲线 | 审稿人1-意见9, 审稿人2-意见3 |
| `exp9_attention_ablation.py` | LBF vs SE/AttGate/CBAM | 审稿人2-意见3 |
| `exp11_spatial_analysis.py` | 电阻抗阳性点空间一致性分析 | 审稿人1-意见14 |
| `exp12_aug_ablation.py` | 数据增强消融（损失曲线） | 审稿人1-意见13 |
| `train_updated.py` | 更新版训练脚本（集成实验8/4钩子） | — |

---

## 快速开始

### 环境准备
```bash
pip install scipy scikit-learn thop torchinfo
pip install ultralytics              # 实验6 YOLOv8
pip install segment-anything          # 实验6 MedSAM
pip install segmentation_models_pytorch  # 实验3 对比模型
```

---

## 各实验使用说明

### 实验1：5折交叉验证
```bash
python exp1_kfold_cv.py
```
**输出：**
- `cv_results/cv_summary.csv`：各种子的均值±标准差
- `cv_results/cv_detail.csv`：每折详细指标

**配置修改：**
- `CFG["VOCdevkit_path"]`：数据集路径
- `TAU_VALUES`、`SEEDS`：可调整

---

### 实验2：Wilcoxon检验
**前提：** 所有对比模型需先运行预测，将预测 PNG 存入：
```
/dataset/zhuluoji/unet/wilcoxon_preds/
  UNet/         ← 每张图一个 PNG（文件名=image_id.png）
  UNetPP/
  RD-SACSeg/
  ...
```
```bash
python exp2_wilcoxon_test.py
```
**输出：**
- `wilcoxon_results/wilcoxon_results.csv`：p值、效应量
- `wilcoxon_results/per_image_metrics.csv`：逐图IoU/Dice

---

### 实验3：效率指标
```bash
python exp3_efficiency.py
```
- 自动检测已安装的模型库
- 输出 `efficiency_results/efficiency_metrics.csv`

**若 thop 未安装：**
```bash
pip install thop
```

---

### 实验4：ROC曲线
**步骤1：** 先用训练好的各模型生成概率图（`.npy`格式）
- 在 `train_updated.py` 最后几个epoch会自动保存
- 或手动调用 `exp4_roc_auc.py` 中的 `save_prob_maps()`

**步骤2：** 将各模型概率图整理到：
```
/dataset/zhuluoji/unet/prob_maps/
  RD-SACSeg/    ← 每张图一个 .npy 文件
  UNet/
  ...
```

**步骤3：**
```bash
python exp4_roc_auc.py
```
**输出：** `roc_results/roc_pr_curves.pdf`

---

### 实验5：Fisher检验
**输入数据格式（`clinical_data.csv`）：**
```
subject_id, group, acupoint, positive
失眠焦虑组_001, 失眠焦虑组, 内关, 1
...
```
若文件不存在，会自动生成演示数据。

```bash
python exp5_fisher_clinical.py
```
**输出：** `fisher_results/fisher_results.csv`

---

### 实验6：新对比模型
**修改文件头部的权重路径：**
```python
MEDSAM_CHECKPOINT  = "/your/path/medsam_vit_b.pth"
VMAMBA_CHECKPOINT  = "/your/path/vmamba_seg.pth"
YOLOV8_CHECKPOINT  = "/your/path/yolov8n-seg.pt"
```
```bash
python exp6_new_models.py
```

---

### 实验7：LBF超参数消融
```bash
python exp7_lbf_ablation.py
```
- 自动测试 γ ∈ {1, 5, 10, 20, 50}
- 自动测试 α/β 初始化策略
- 输出 `ablation_lbf/lbf_ablation.csv`

---

### 实验8：LBF收敛曲线
**集成到训练：** 使用 `train_updated.py` 替换原 `train.py`
- 训练过程中自动记录参数
- 训练结束后自动生成收敛曲线

**单独绘图（已有日志时）：**
```bash
python exp8_lbf_convergence.py
```
**输出：**
- `lbf_convergence/gradient_derivation.txt`：LaTeX梯度公式
- `lbf_convergence/*.pdf`：α/β/tau收敛曲线

---

### 实验9：注意力机制消融
```bash
python exp9_attention_ablation.py
```
- 依次训练 SE/AttGate/CBAM/LBF
- 输出 `ablation_attention/attention_ablation.csv`

---

### 实验11：空间一致性分析
**准备数据：**
```
clinical/seg_results/   ← 分割结果 PNG（patient_id.png）
clinical/acupoint_masks/ ← 穴区掩码 PNG
clinical/points/         ← 阳性点 JSON
subject_list.json        ← 受试者列表
```
若无真实数据，自动生成演示数据。
```bash
python exp11_spatial_analysis.py
```
**输出：**
- `spatial_analysis/group_spatial_summary.csv`
- `spatial_analysis/spatial_analysis.pdf`（箱线图）

---

### 实验12：增强消融
```bash
python exp12_aug_ablation.py
```
- 分别训练有/无增强两组
- 输出损失曲线和IoU提升幅度
- `ablation_augmentation/aug_loss_curves.pdf`

---

## 注意事项

1. **路径统一**：所有脚本中 `VOCdevkit_path`、`GT_DIR` 等路径需统一修改为实际路径
2. **GPU内存**：交叉验证（实验1）和注意力消融（实验9）训练次数多，建议分批运行
3. **数据格式**：
   - 预测掩码：`uint8`，像素值=类别索引（0/1）
   - 概率图：`float32`，值域 [0,1]，`.npy` 格式
4. **实验复现性**：所有实验均使用固定随机种子，结果可复现
5. **LaTeX输出**：每个脚本末尾都有 LaTeX 表格片段，可直接粘贴到论文

---

## 输出目录结构
```
项目根目录/
  cv_results/           # 实验1
  wilcoxon_results/     # 实验2
  efficiency_results/   # 实验3
  roc_results/          # 实验4
  fisher_results/       # 实验5
  exp6_new_models/      # 实验6
  ablation_lbf/         # 实验7
  lbf_convergence/      # 实验8
  ablation_attention/   # 实验9
  spatial_analysis/     # 实验11
  ablation_augmentation/# 实验12
```
