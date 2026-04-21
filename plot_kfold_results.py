"""
plot_kfold_results.py
---------------------
读取 kfold_results.json，生成：
  1. 实验A：各折 mIoU / Dice 柱状图（均值±标准差横线）
  2. 实验B：各种子折均值折线图，展示跨种子稳定性
用法：
    python plot_kfold_results.py --json path/to/kfold_results.json
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_experiment_A(result_A, out_dir):
    per_fold  = result_A['per_fold']
    summary   = result_A['summary']
    n_folds   = len(per_fold)
    folds_x   = list(range(1, n_folds + 1))

    mious = [r['mIoU'] * 100 for r in per_fold]
    dices = [r['Dice']  * 100 for r in per_fold]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Experiment A: 5-Fold Cross Validation (fixed seed)', fontsize=14)

    for ax, values, metric_name, color, summary_key_mean, summary_key_std in [
        (axes[0], mious, 'mIoU (%)',  'steelblue', 'mIoU_mean', 'mIoU_std'),
        (axes[1], dices, 'Dice (%)',  'darkorange','Dice_mean',  'Dice_std'),
    ]:
        bars = ax.bar(folds_x, values, color=color, alpha=0.75, edgecolor='black', width=0.5)
        mean_val = summary[summary_key_mean] * 100
        std_val  = summary[summary_key_std]  * 100
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean={mean_val:.2f}%')
        ax.fill_between([0.5, n_folds + 0.5],
                        mean_val - std_val, mean_val + std_val,
                        color='red', alpha=0.10, label=f'±Std={std_val:.2f}%')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12)
        ax.set_xticks(folds_x)
        ax.legend(fontsize=9)
        y_min = max(0, min(values) - 3)
        y_max = min(100, max(values) + 3)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'experiment_A_kfold.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_experiment_B(result_B, out_dir):
    per_seed  = result_B['per_seed_summary']
    n_seeds   = len(per_seed)

    miou_means = [s['mIoU_mean'] * 100 for s in per_seed]
    miou_stds  = [s['mIoU_std']  * 100 for s in per_seed]
    dice_means = [s['Dice_mean'] * 100 for s in per_seed]
    dice_stds  = [s['Dice_std']  * 100 for s in per_seed]

    x = list(range(1, n_seeds + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Experiment B: Multi-Seed Stability (5-fold CV per seed)', fontsize=14)

    for ax, means, stds, metric_name, color, cross_mean_key, cross_std_key in [
        (axes[0], miou_means, miou_stds, 'mIoU (%)',  'steelblue',
         'cross_seed_mIoU_mean', 'cross_seed_mIoU_std'),
        (axes[1], dice_means, dice_stds, 'Dice (%)',  'darkorange',
         'cross_seed_Dice_mean', 'cross_seed_Dice_std'),
    ]:
        ax.errorbar(x, means, yerr=stds, fmt='o-', color=color, capsize=5,
                    linewidth=1.5, markersize=6, label='Seed mean ± fold std')
        cross_mean = result_B[cross_mean_key] * 100
        cross_std  = result_B[cross_std_key]  * 100
        ax.axhline(cross_mean, color='red', linestyle='--', linewidth=1.5,
                   label=f'Cross-seed mean={cross_mean:.2f}%')
        ax.fill_between([0.5, n_seeds + 0.5],
                        cross_mean - cross_std, cross_mean + cross_std,
                        color='red', alpha=0.10, label=f'±Std={cross_std:.2f}%')
        ax.set_xlabel('Seed Index', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12)
        ax.set_xticks(x)
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'experiment_B_multi_seed.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def print_summary_table(data):
    """在终端打印汇总表"""
    print("\n" + "=" * 65)
    print("实验 A — 5折交叉验证（固定种子）")
    print(f"{'Fold':<8} {'mIoU (%)':>12} {'Dice (%)':>12}")
    print("-" * 35)
    for i, r in enumerate(data['experiment_A']['per_fold'], 1):
        print(f"{i:<8} {r['mIoU']*100:>12.2f} {r['Dice']*100:>12.2f}")
    s = data['experiment_A']['summary']
    print("-" * 35)
    print(f"{'Mean':8} {s['mIoU_mean']*100:>12.2f} {s['Dice_mean']*100:>12.2f}")
    print(f"{'Std':8} {s['mIoU_std']*100:>12.2f}  {s['Dice_std']*100:>12.2f}")

    if 'experiment_B' in data:
        print("\n实验 B — 多种子跨折均值")
        print(f"{'Seed#':<8} {'mIoU mean':>12} {'mIoU std':>10} {'Dice mean':>12} {'Dice std':>10}")
        print("-" * 55)
        for i, ss in enumerate(data['experiment_B']['per_seed_summary'], 1):
            print(f"{i:<8} {ss['mIoU_mean']*100:>12.2f} {ss['mIoU_std']*100:>10.2f}"
                  f" {ss['Dice_mean']*100:>12.2f} {ss['Dice_std']*100:>10.2f}")
        print("-" * 55)
        b = data['experiment_B']
        print(f"Cross-seed  mIoU: {b['cross_seed_mIoU_mean']*100:.2f}% ± {b['cross_seed_mIoU_std']*100:.2f}%")
        print(f"Cross-seed  Dice: {b['cross_seed_Dice_mean']*100:.2f}% ± {b['cross_seed_Dice_std']*100:.2f}%")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True, help='kfold_results.json 路径')
    args = parser.parse_args()

    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out_dir = os.path.dirname(args.json)
    print_summary_table(data)

    if 'experiment_A' in data:
        plot_experiment_A(data['experiment_A'], out_dir)
    if 'experiment_B' in data:
        plot_experiment_B(data['experiment_B'], out_dir)
    print("\n可视化图表已保存至:", out_dir)


if __name__ == '__main__':
    main()
