#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import albumentations as A

def create_output_dirs(output_image_dir, output_mask_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

def get_image_mask_pairs(image_dir, mask_dir):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    pairs = []
    for img in image_files:
        # 假设 image 与 mask 同名；也可以加入文件后缀判断
        base = os.path.splitext(img)[0]
        # 在 mask_dir 内寻找同名文件（后缀不一定一样，所以遍历比较 basename）
        found = False
        for m in mask_files:
            if os.path.splitext(m)[0] == base:
                pairs.append((
                    os.path.join(image_dir, img),
                    os.path.join(mask_dir, m)
                ))
                found = True
                break
        if not found:
            print(f"警告：在 {mask_dir} 中找不到与 {img} 对应的 mask 文件。")
    return pairs

def apply_mixup(img1, mask1, img2, mask2):
    """MixUp数据增强实现（自动统一尺寸）"""
    target_h, target_w = img1.shape[:2]
    img2 = cv2.resize(img2, (target_w, target_h))
    mask2 = cv2.resize(mask2, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    if len(img1.shape) != len(img2.shape):
        if len(img1.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    alpha = random.uniform(0.3, 0.7)
    mixed_img = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    mixed_mask = cv2.addWeighted(mask1, alpha, mask2, 1 - alpha, 0)
    return mixed_img, mixed_mask

def apply_mosaic(images, masks, original_size=(512, 512)):
    h, w = original_size  # 原始图片尺寸
    mosaic_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    mosaic_mask = np.zeros((h * 2, w * 2), dtype=np.uint8)
    for i in range(4):
        img = images[i]
        mask = masks[i]
        scale = random.uniform(0.5, 1.5)
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h < h or new_w < w:
            scale = max(h / img.shape[0], w / img.shape[1])
            new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
        img = cv2.resize(img, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        x_max = max(0, new_w - w)
        y_max = max(0, new_h - h)
        x = random.randint(0, x_max) if x_max > 0 else 0
        y = random.randint(0, y_max) if y_max > 0 else 0
        if new_h >= h and new_w >= w:
            img_crop = img[y:y + h, x:x + w]
            mask_crop = mask[y:y + h, x:x + w]
        else:
            img_crop = cv2.resize(img, (w, h))
            mask_crop = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if i == 0:
            mosaic_img[:h, :w] = img_crop
            mosaic_mask[:h, :w] = mask_crop
        elif i == 1:
            mosaic_img[:h, w:] = img_crop
            mosaic_mask[:h, w:] = mask_crop
        elif i == 2:
            mosaic_img[h:, :w] = img_crop
            mosaic_mask[h:, :w] = mask_crop
        else:
            mosaic_img[h:, w:] = img_crop
            mosaic_mask[h:, w:] = mask_crop
    return (cv2.resize(mosaic_img, (w, h)),
            cv2.resize(mosaic_mask, (w, h), interpolation=cv2.INTER_NEAREST))

def augment_data(pairs, output_image_dir, output_mask_dir, augmentations, mixup_count=2, mosaic_count=2):
    """
    1. 每个图像先对单一变换进行增强，并保存结果；  
    2. 然后用完整增强管道（随机组合）进行一次增强；  
    3. 接着用 mixup 和 mosaic 分别进行增强。
    """
    # 预加载所有图片及对应的 mask
    loaded_pairs = []
    for img_path, mask_path in pairs:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        loaded_pairs.append((img, mask))
    
    all_images = [p[0] for p in loaded_pairs]
    all_masks = [p[1] for p in loaded_pairs]

    for idx in tqdm(range(len(loaded_pairs)), desc="Augmenting"):
        img, mask = loaded_pairs[idx]
        base_filename = os.path.splitext(os.path.basename(pairs[idx][0]))[0]
        
        # 【可选】保存原始图片及 mask（如果需要保留）
        cv2.imwrite(os.path.join(output_image_dir, f"{base_filename}_orig.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_mask_dir, f"{base_filename}_orig.png"), mask)
        
        # 1. 分别对每个单独 transform 执行增强
        for t_idx, transform in enumerate(augmentations.transforms):
            single_aug = A.Compose([transform], additional_targets={'mask': 'mask'})
            augmented = single_aug(image=img.copy(), mask=mask.copy(), force_apply=True)
            aug_img = augmented['image']
            aug_mask = augmented['mask']
            out_img_path = os.path.join(output_image_dir, f"{base_filename}_trans_{t_idx}.png")
            out_mask_path = os.path.join(output_mask_dir, f"{base_filename}_trans_{t_idx}.png")
            cv2.imwrite(out_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_mask_path, aug_mask)
        
        # 2. 使用完整的增强管道进行一次增强
        augmented = augmentations(image=img.copy(), mask=mask.copy())
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        out_img_path = os.path.join(output_image_dir, f"{base_filename}_full.png")
        out_mask_path = os.path.join(output_mask_dir, f"{base_filename}_full.png")
        cv2.imwrite(out_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_mask_path, aug_mask)
        
        # 3. 进行 mixup 增强 mixup_count 次
        for i in range(mixup_count):
            other_idx = random.randint(0, len(loaded_pairs) - 1)
            other_img = all_images[other_idx]
            other_mask = all_masks[other_idx]
            mixup_img, mixup_mask = apply_mixup(img.copy(), mask.copy(), other_img.copy(), other_mask.copy())
            out_img_path = os.path.join(output_image_dir, f"{base_filename}_mixup_{i}.png")
            out_mask_path = os.path.join(output_mask_dir, f"{base_filename}_mixup_{i}.png")
            cv2.imwrite(out_img_path, cv2.cvtColor(mixup_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_mask_path, mixup_mask)
        
        # 4. 进行 mosaic 增强 mosaic_count 次
        if len(loaded_pairs) >= 4:
            for i in range(mosaic_count):
                indices = [idx]
                other_indices = list(range(len(loaded_pairs)))
                other_indices.remove(idx)
                indices.extend(random.sample(other_indices, 3))
                mosaic_imgs = [all_images[i] for i in indices]
                mosaic_masks = [all_masks[i] for i in indices]
                mosaic_img, mosaic_mask = apply_mosaic(mosaic_imgs, mosaic_masks)
                out_img_path = os.path.join(output_image_dir, f"{base_filename}_mosaic_{i}.png")
                out_mask_path = os.path.join(output_mask_dir, f"{base_filename}_mosaic_{i}.png")
                cv2.imwrite(out_img_path, cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(out_mask_path, mosaic_mask)

def main():
    # original_image_dir 为原始图片文件夹路径
    original_image_dir = "/dataset/zhuluoji/unet/unet_color/VOCdevkit/VOC2007/JPEGImages"
    
    # 自动构造与 original_image_dir 同级目录下的 mask_output_folder 路径  
    # 注意：这里假设 mask_output_folder 与 original_image_dir 处于同一父目录下  
    parent_dir = os.path.dirname(original_image_dir)
    original_mask_dir = os.path.join(parent_dir, "mask_output_folder")

    # 构造 mask_folder 和 overlay_folder 路径
    augmented_image_dir = os.path.join(parent_dir, "JPEGImages_Aug")
    augmented_mask_dir = os.path.join(parent_dir, "SegmentationClass_Aug")
    # # 输出增强结果的目标文件夹（你也可以根据需要修改位置）
    # augmented_image_dir = "/dataset/zhuluoji/unet/unet-pytorch-main/VOCdevkit/VOC2007/JPEGImages"
    # augmented_mask_dir = "/dataset/zhuluoji/unet/unet-pytorch-main/VOCdevkit/VOC2007/SegmentationClass"

    # 创建输出目录
    create_output_dirs(augmented_image_dir, augmented_mask_dir)
    
    pairs = get_image_mask_pairs(original_image_dir, original_mask_dir)
    print(f"共找到 {len(pairs)} 对 image-mask 文件。")

    # 构造增强管道：多个 transform 既可单独使用，也可组合为完整链
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT,
            p=0.8
        ),
        A.Affine(
            shear=(-15, 15),
            translate_percent=(-0.1, 0.1),
            scale=(0.8, 1.2),
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.3),
            contrast_limit=(-0.3, 0.3),
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.2
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.RandomSunFlare(
            src_radius=100,
            num_flare_circles_lower=3,
            num_flare_circles_upper=6,
            p=0.1
        ),
        A.RandomShadow(
            num_shadows_lower=1,
            num_shadows_upper=3,
            shadow_dimension=5,
            p=0.1
        ),
        A.PixelDropout(dropout_prob=0.01, p=0.1),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.3
        )
    ], additional_targets={'mask': 'mask'})

    # 混合增强次数（mixup 和 mosaic 可根据需要调整）
    augment_data(pairs, augmented_image_dir, augmented_mask_dir, augmentations,
                 mixup_count=2, mosaic_count=2)
    print("数据增强完成！")

if __name__ == "__main__":
    main()