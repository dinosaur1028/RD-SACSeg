import datetime
import os
import json
import random

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
os.chdir(f'{parent_directory}')

from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

net_name = 'unet'
from nets.unet_final import Unet
from plot_p import ParameterLogger

from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch
from utils.utils_metrics import compute_all
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ============================================================
# 工具函数
# ============================================================

def split_kfold(all_lines, n_splits=5, seed=42):
    """将数据集划分为 n_splits 折，返回 [(train_lines, val_lines), ...] 列表"""
    rng = random.Random(seed)
    indices = list(range(len(all_lines)))
    rng.shuffle(indices)
    fold_size = len(indices) // n_splits
    folds = []
    for k in range(n_splits):
        val_start = k * fold_size
        val_end = val_start + fold_size if k < n_splits - 1 else len(indices)
        val_idx = set(indices[val_start:val_end])
        train_lines = [all_lines[i] for i in range(len(all_lines)) if i not in val_idx]
        val_lines   = [all_lines[i] for i in range(len(all_lines)) if i in val_idx]
        folds.append((train_lines, val_lines))
    return folds


def build_model(num_classes, backbone, pretrained, model_path, device, local_rank=0):
    """构建并加载模型权重，返回 model"""
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("Successful Load Key Num:", len(load_key))
            print("Fail To Load Key Num:", len(no_load_key))
    return model


def train_one_fold(
    fold_idx,
    train_lines, val_lines,
    cfg,          # dict，存放所有超参
    save_dir,
    device,
    seed,
):
    """训练单折，返回该折的最佳 val_loss 及 metrics 字典"""
    local_rank = 0
    seed_everything(seed)

    # ---------- 模型 ----------
    model = build_model(
        cfg['num_classes'], cfg['backbone'], cfg['pretrained'],
        cfg['model_path'], device, local_rank
    )

    # ---------- Loss 历史 ----------
    time_str   = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir    = os.path.join(save_dir, f"fold{fold_idx}_loss_{time_str}")
    loss_history = LossHistory(log_dir, model, input_shape=cfg['input_shape'])

    # ---------- fp16 ----------
    scaler = None
    if cfg['fp16']:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    # ---------- 模型放到GPU ----------
    model_train = model.train()
    if cfg['Cuda']:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # ---------- 数据集 ----------
    num_train = len(train_lines)
    num_val   = len(val_lines)

    # ---------- 冻结 / 解冻训练循环 ----------
    UnFreeze_flag = False
    if cfg['Freeze_Train']:
        model.freeze_backbone()

    batch_size = cfg['Freeze_batch_size'] if cfg['Freeze_Train'] else cfg['Unfreeze_batch_size']

    nbs          = 16
    lr_limit_max = 1e-4 if cfg['optimizer_type'] == 'adam' else 1e-1
    lr_limit_min = 1e-4 if cfg['optimizer_type'] == 'adam' else 5e-4
    Init_lr_fit  = min(max(batch_size / nbs * cfg['Init_lr'], lr_limit_min), lr_limit_max)
    Min_lr_fit   = min(max(batch_size / nbs * cfg['Min_lr'], lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit,
                           betas=(cfg['momentum'], 0.999), weight_decay=cfg['weight_decay']),
        'sgd':  optim.SGD(model.parameters(), Init_lr_fit,
                          momentum=cfg['momentum'], nesterov=True, weight_decay=cfg['weight_decay'])
    }[cfg['optimizer_type']]

    lr_scheduler_func = get_lr_scheduler(
        cfg['lr_decay_type'], Init_lr_fit, Min_lr_fit, cfg['UnFreeze_Epoch']
    )

    train_dataset = UnetDataset(train_lines, cfg['input_shape'], cfg['num_classes'], True,  cfg['VOCdevkit_path'])
    val_dataset   = UnetDataset(val_lines,   cfg['input_shape'], cfg['num_classes'], False, cfg['VOCdevkit_path'])

    gen = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size,
        num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
        collate_fn=unet_dataset_collate,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=seed)
    )
    gen_val = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size,
        num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
        collate_fn=unet_dataset_collate,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=seed)
    )

    eval_callback = EvalCallback(
        model, cfg['input_shape'], cfg['num_classes'], val_lines,
        cfg['VOCdevkit_path'], log_dir, cfg['Cuda'],
        eval_flag=cfg['eval_flag'], period=cfg['eval_period']
    )

    # ---------- 主训练循环 ----------
    for epoch in range(cfg['Init_Epoch'], cfg['UnFreeze_Epoch']):
        # 解冻
        if epoch >= cfg['Freeze_Epoch'] and not UnFreeze_flag and cfg['Freeze_Train']:
            batch_size = cfg['Unfreeze_batch_size']
            Init_lr_fit = min(max(batch_size / nbs * cfg['Init_lr'], lr_limit_min), lr_limit_max)
            Min_lr_fit  = min(max(batch_size / nbs * cfg['Min_lr'], lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(
                cfg['lr_decay_type'], Init_lr_fit, Min_lr_fit, cfg['UnFreeze_Epoch']
            )
            model.unfreeze_backbone()
            gen = DataLoader(
                train_dataset, shuffle=True, batch_size=batch_size,
                num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
                collate_fn=unet_dataset_collate,
                worker_init_fn=partial(worker_init_fn, rank=0, seed=seed)
            )
            gen_val = DataLoader(
                val_dataset, shuffle=False, batch_size=batch_size,
                num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
                collate_fn=unet_dataset_collate,
                worker_init_fn=partial(worker_init_fn, rank=0, seed=seed)
            )
            UnFreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        epoch_step     = num_train // batch_size
        epoch_step_val = num_val   // batch_size

        fit_one_epoch(
            model_train, model, loss_history, eval_callback, optimizer, epoch,
            epoch_step, epoch_step_val, gen, gen_val,
            cfg['UnFreeze_Epoch'], cfg['Cuda'],
            cfg['dice_loss'], cfg['focal_loss'], cfg['cls_weights'],
            cfg['num_classes'], cfg['fp16'], scaler,
            cfg['save_period'], save_dir, local_rank
        )

    loss_history.writer.close()

    # ---------- 读取该折最终指标（从 eval_callback 的 mIoU 日志中）----------
    metrics = _read_fold_best_metrics(log_dir, cfg['num_classes'])
    return metrics


def _read_fold_best_metrics(log_dir, num_classes):
    """
    尝试从 EvalCallback 保存的 epoch_miou.txt 中读取最优指标。
    若文件不存在，返回 None（后续可改为读取 confusion matrix 计算）。
    """
    miou_path = os.path.join(log_dir, 'epoch_miou.txt')
    if not os.path.exists(miou_path):
        # 兼容旧版：尝试 miou_*.txt
        candidates = [f for f in os.listdir(log_dir) if f.startswith('epoch_miou')]
        if candidates:
            miou_path = os.path.join(log_dir, candidates[0])
        else:
            print(f"  [警告] 未找到 mIoU 日志文件于 {log_dir}，该折指标将置 NaN")
            return {'mIoU': float('nan'), 'Dice': float('nan')}

    values = []
    with open(miou_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    pass
    if not values:
        return {'mIoU': float('nan'), 'Dice': float('nan')}

    best_miou = max(values)
    # Dice ≈ 2*IoU/(1+IoU) for binary （近似估计，若需精确值请用 compute_all 在测试集上单独推断）
    best_dice = 2 * best_miou / (1 + best_miou)
    return {'mIoU': best_miou, 'Dice': best_dice}


def summarize_results(all_results, tag=""):
    """打印均值 ± 标准差"""
    mious = [r['mIoU'] for r in all_results if not np.isnan(r['mIoU'])]
    dices = [r['Dice'] for r in all_results if not np.isnan(r['Dice'])]
    print(f"\n{'='*60}")
    print(f"[{tag}] 汇总结果 ({len(mious)} 折有效)")
    print(f"  mIoU : {np.mean(mious)*100:.2f}% ± {np.std(mious)*100:.2f}%")
    print(f"  Dice : {np.mean(dices)*100:.2f}% ± {np.std(dices)*100:.2f}%")
    print(f"{'='*60}\n")
    return {
        'mIoU_mean': float(np.mean(mious)),
        'mIoU_std':  float(np.std(mious)),
        'Dice_mean': float(np.mean(dices)),
        'Dice_std':  float(np.std(dices)),
    }


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":

    # ---- 基础配置（与原 train.py 保持一致）----
    Cuda              = True
    base_seed         = 11          # 主随机种子（5折时使用）
    multi_seeds       = [11, 42, 123, 2024, 9999]   # 多随机种子实验
    n_splits          = 5           # K折数量
    run_multi_seed    = True        # 是否运行多随机种子实验

    num_classes       = 2
    backbone          = "vgg"
    pretrained        = False
    model_path        = "/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-11-05 16:53:48/best_epoch_weights.pth"
    input_shape       = [512, 512]

    Init_Epoch        = 0
    Freeze_Epoch      = 50
    Freeze_batch_size = 2
    UnFreeze_Epoch    = 300
    Unfreeze_batch_size = 2
    Freeze_Train      = True

    Init_lr           = 1e-4
    Min_lr            = Init_lr * 0.01
    optimizer_type    = "adam"
    momentum          = 0.9
    weight_decay      = 0
    lr_decay_type     = 'cos'
    save_period       = 10

    eval_flag         = True
    eval_period       = 5
    fp16              = False

    VOCdevkit_path    = '/dataset/zhuluoji/unet/VOCdevkit_dilate'
    dice_loss         = True
    focal_loss        = True
    cls_weights       = np.ones([num_classes], np.float32)
    num_workers       = 4

    # ---- 读取全量数据 ----
    all_lines_path = os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation")
    train_txt = os.path.join(all_lines_path, "train.txt")
    val_txt   = os.path.join(all_lines_path, "val.txt")
    with open(train_txt, "r", encoding="gbk") as f:
        train_lines_orig = f.readlines()
    with open(val_txt, "r", encoding="gbk") as f:
        val_lines_orig = f.readlines()
    all_lines = train_lines_orig + val_lines_orig   # 合并后重新划分
    print(f"总样本数: {len(all_lines)}")

    # ---- 设备 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 超参打包 ----
    cfg = dict(
        num_classes=num_classes, backbone=backbone, pretrained=pretrained,
        model_path=model_path, input_shape=input_shape,
        Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
        Freeze_Train=Freeze_Train,
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type,
        momentum=momentum, weight_decay=weight_decay, lr_decay_type=lr_decay_type,
        save_period=save_period, eval_flag=eval_flag, eval_period=eval_period,
        fp16=fp16, Cuda=Cuda, VOCdevkit_path=VOCdevkit_path,
        dice_loss=dice_loss, focal_loss=focal_loss, cls_weights=cls_weights,
        num_workers=num_workers,
    )

    # ========================================================
    # 实验 A：固定种子的 5 折交叉验证
    # ========================================================
    now_str   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir  = f'train_weight_output/kfold_{now_str}'
    os.makedirs(base_dir, exist_ok=True)

    # 备份本文件
    shutil.copy2(os.path.abspath(__file__), base_dir)

    print(f"\n{'='*60}")
    print(f"实验 A：5折交叉验证（seed={base_seed}）")
    print(f"{'='*60}")

    folds = split_kfold(all_lines, n_splits=n_splits, seed=base_seed)
    fold_results_A = []

    for k, (tr, vl) in enumerate(folds):
        print(f"\n---- Fold {k+1}/{n_splits}  train={len(tr)}  val={len(vl)} ----")
        fold_save_dir = os.path.join(base_dir, f"seed{base_seed}_fold{k+1}")
        os.makedirs(fold_save_dir, exist_ok=True)

        metrics = train_one_fold(
            fold_idx=k+1,
            train_lines=tr, val_lines=vl,
            cfg=cfg,
            save_dir=fold_save_dir,
            device=device,
            seed=base_seed + k,   # 每折内部用不同种子，保证可复现
        )
        fold_results_A.append(metrics)
        print(f"  Fold {k+1} => mIoU={metrics['mIoU']*100:.2f}%  Dice={metrics['Dice']*100:.2f}%")

    summary_A = summarize_results(fold_results_A, tag=f"5折CV seed={base_seed}")

    # ========================================================
    # 实验 B：5 个不同随机种子的 5 折交叉验证（报告跨种子标准差）
    # ========================================================
    if run_multi_seed:
        print(f"\n{'='*60}")
        print(f"实验 B：多随机种子实验（{len(multi_seeds)} 个种子，每个 5 折）")
        print(f"{'='*60}")

        seed_summaries = []   # 每个种子对应一个 (mIoU_mean, Dice_mean)

        for seed in multi_seeds:
            print(f"\n==== Seed {seed} ====")
            folds_s = split_kfold(all_lines, n_splits=n_splits, seed=seed)
            seed_fold_results = []

            for k, (tr, vl) in enumerate(folds_s):
                print(f"  -- Seed{seed} Fold {k+1}/{n_splits}  train={len(tr)}  val={len(vl)} --")
                fold_save_dir = os.path.join(base_dir, f"seed{seed}_fold{k+1}")
                os.makedirs(fold_save_dir, exist_ok=True)

                metrics = train_one_fold(
                    fold_idx=k+1,
                    train_lines=tr, val_lines=vl,
                    cfg=cfg,
                    save_dir=fold_save_dir,
                    device=device,
                    seed=seed + k,
                )
                seed_fold_results.append(metrics)
                print(f"  Seed{seed} Fold{k+1} => mIoU={metrics['mIoU']*100:.2f}%  Dice={metrics['Dice']*100:.2f}%")

            seed_summary = summarize_results(seed_fold_results, tag=f"seed={seed}")
            seed_summaries.append(seed_summary)

        # 跨种子统计
        all_miou_means = [s['mIoU_mean'] for s in seed_summaries]
        all_dice_means = [s['Dice_mean'] for s in seed_summaries]
        print(f"\n{'='*60}")
        print(f"[实验 B 汇总] 跨 {len(multi_seeds)} 个随机种子")
        print(f"  mIoU: {np.mean(all_miou_means)*100:.2f}% ± {np.std(all_miou_means)*100:.2f}%")
        print(f"  Dice: {np.mean(all_dice_means)*100:.2f}% ± {np.std(all_dice_means)*100:.2f}%")
        print(f"{'='*60}\n")
    else:
        seed_summaries = []

    # ========================================================
    # 保存汇总 JSON
    # ========================================================
    result_json = {
        'experiment_A': {
            'description': f'5折CV，固定seed={base_seed}',
            'per_fold': fold_results_A,
            'summary': summary_A,
        },
    }
    if run_multi_seed:
        result_json['experiment_B'] = {
            'description': f'{len(multi_seeds)}个随机种子各做5折CV',
            'per_seed_summary': seed_summaries,
            'cross_seed_mIoU_mean': float(np.mean(all_miou_means)),
            'cross_seed_mIoU_std':  float(np.std(all_miou_means)),
            'cross_seed_Dice_mean': float(np.mean(all_dice_means)),
            'cross_seed_Dice_std':  float(np.std(all_dice_means)),
        }

    json_path = os.path.join(base_dir, 'kfold_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"汇总结果已保存到: {json_path}")
