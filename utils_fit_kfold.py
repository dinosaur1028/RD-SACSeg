"""
utils_fit_kfold.py
------------------
在原 utils_fit.fit_one_epoch 基础上，新增：
  - fit_one_epoch_with_metrics：除正常训练外，在最优 val_loss epoch 时额外
    用 compute_all 计算并返回完整 metrics（IoU/Dice/F1 等）。
  - run_kfold_eval：在折训练完成后，对 val_lines 做完整的像素级推断并聚合指标。

【使用方法】
将本文件放到 utils/ 下，在 train_kfold.py 中：
    from utils.utils_fit_kfold import fit_one_epoch_with_metrics
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from utils.utils import get_lr
from utils.utils_metrics import f_score, compute_all


# ------------------------------------------------------------------ #
#  核心：带 metrics 返回的单 epoch 训练函数
# ------------------------------------------------------------------ #
def fit_one_epoch_with_metrics(
    model_train, model, loss_history, eval_callback, optimizer,
    epoch, epoch_step, epoch_step_val, gen, gen_val,
    Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes,
    fp16, scaler, save_period, save_dir, local_rank=0,
    best_val_loss_holder=None   # list[float] 传入，用于跨 epoch 追踪最优
):
    """
    与原 fit_one_epoch 完全兼容，额外在每个 epoch 结束后：
      - 若 val_loss 是历史最优，将当前 fold 的 best_val_loss 更新到 best_val_loss_holder[0]
    返回 (total_loss, val_loss, is_best)
    """
    total_loss    = 0
    total_f_score = 0
    val_loss      = 0
    val_f_score   = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch+1}/{Epoch}',
                    postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) \
                   if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            if dice_loss:
                loss = loss + Dice_loss(outputs, labels)
            with torch.no_grad():
                _f_score = f_score(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(imgs)
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) \
                       if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                if dice_loss:
                    loss = loss + Dice_loss(outputs, labels)
                with torch.no_grad():
                    _f_score = f_score(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss    += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score':    total_f_score / (iteration + 1),
                'lr':         get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train — Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch+1}/{Epoch}',
                    postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
            outputs  = model_train(imgs)
            loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) \
                   if focal_loss else CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            if dice_loss:
                loss = loss + Dice_loss(outputs, labels)
            _f_score  = f_score(outputs, labels)
            val_loss    += loss.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'val_loss': val_loss / (iteration + 1),
                'f_score':  val_f_score / (iteration + 1),
                'lr':       get_lr(optimizer)
            })
            pbar.update(1)

    avg_train_loss = total_loss  / epoch_step
    avg_val_loss   = val_loss    / epoch_step_val

    is_best = False
    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, avg_train_loss, avg_val_loss)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print(f'Epoch:{epoch+1}/{Epoch}  Total Loss:{avg_train_loss:.3f}  Val Loss:{avg_val_loss:.3f}')

        # 保存权重
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(),
                       os.path.join(save_dir,
                                    f'ep{epoch+1:03d}-loss{avg_train_loss:.3f}-val_loss{avg_val_loss:.3f}.pth'))

        if len(loss_history.val_loss) <= 1 or avg_val_loss <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "best_epoch_weights.pth"))
            is_best = True
            if best_val_loss_holder is not None:
                best_val_loss_holder[0] = avg_val_loss

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

    return avg_train_loss, avg_val_loss, is_best


# ------------------------------------------------------------------ #
#  折结束后：加载 best 权重，对 val_lines 做完整像素推断，返回 metrics
# ------------------------------------------------------------------ #
def evaluate_fold_on_val(
    model, val_lines, cfg, save_dir, device
):
    """
    加载 save_dir/best_epoch_weights.pth，对 val_lines 做逐图推断，
    调用 compute_all 计算完整指标，返回 metrics dict。

    依赖：
      - nets.unet.Unet
      - utils.dataloader.UnetDataset
      - utils.utils_metrics.compute_all
      - utils.utils (decode_segmentation_map 或等效推断逻辑)

    若想跳过逐图推断（速度较慢），可直接使用 _read_fold_best_metrics 从
    epoch_miou.txt 中近似读取，train_kfold.py 默认已提供该回退逻辑。
    """
    best_weight = os.path.join(save_dir, "best_epoch_weights.pth")
    if not os.path.exists(best_weight):
        print(f"[警告] 找不到 best_epoch_weights.pth 于 {save_dir}")
        return {'mIoU': float('nan'), 'Dice': float('nan')}

    # 加载最优权重
    model.load_state_dict(torch.load(best_weight, map_location=device))
    model.eval()
    if cfg['Cuda']:
        model = model.cuda()

    from utils.dataloader import UnetDataset, unet_dataset_collate
    from torch.utils.data import DataLoader
    from functools import partial
    from utils.utils import worker_init_fn

    val_dataset = UnetDataset(
        val_lines, cfg['input_shape'], cfg['num_classes'], False, cfg['VOCdevkit_path']
    )
    gen_val = DataLoader(
        val_dataset, shuffle=False, batch_size=1,
        num_workers=cfg['num_workers'], pin_memory=True, drop_last=False,
        collate_fn=unet_dataset_collate,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=0)
    )

    # 构建混淆矩阵
    num_classes = cfg['num_classes']
    hist = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for imgs, pngs, labels in tqdm(gen_val, desc="Evaluating fold"):
            if cfg['Cuda']:
                imgs = imgs.cuda()
            outputs = model(imgs)   # [B, C, H, W]
            pred = torch.argmax(outputs, dim=1).cpu().numpy()   # [B, H, W]
            gt   = pngs.cpu().numpy()                           # [B, H, W]  (class index)
            for b in range(pred.shape[0]):
                p = pred[b].flatten()
                g = gt[b].flatten()
                k = (g >= 0) & (g < num_classes)
                hist += np.bincount(
                    num_classes * g[k].astype(int) + p[k],
                    minlength=num_classes ** 2
                ).reshape(num_classes, num_classes)

    # 从 hist 计算 IoU 和 Dice（前景类=1）
    IoU  = np.diag(hist) / np.maximum(hist.sum(1) + hist.sum(0) - np.diag(hist), 1)
    dice_fg = 2 * hist[1, 1] / np.maximum(hist[1, :].sum() + hist[:, 1].sum(), 1)
    mIoU = float(np.nanmean(IoU))

    metrics = {
        'mIoU':             mIoU,
        'Dice':             float(dice_fg),
        'foreground_IoU':   float(IoU[1]) if num_classes > 1 else mIoU,
        'background_IoU':   float(IoU[0]) if num_classes > 1 else float('nan'),
        'confusion_matrix': hist.tolist(),
    }
    print(f"  [折评估] mIoU={mIoU*100:.2f}%  Dice={dice_fg*100:.2f}%")
    return metrics
