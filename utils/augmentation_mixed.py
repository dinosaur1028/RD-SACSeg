#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
SEED = 42
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 提取当前文件所在的父级目录路径
pparent_directory = os.path.dirname(os.path.dirname(current_file_path))
def create_output_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_image_quadruple_pairs(image_dir, seg_mask_dir, dex_mask_dir, lab_mask_dir):
    image_files = sorted(os.listdir(image_dir))
    seg_files = sorted(os.listdir(seg_mask_dir))
    dex_files = sorted(os.listdir(dex_mask_dir))
    lab_files = sorted(os.listdir(lab_mask_dir))
    pairs = []
    for img in image_files:
        base = os.path.splitext(img)[0]
        seg_found = None
        dex_found = None
        lab_found = None
        for m in seg_files:
            if os.path.splitext(m)[0] == base:
                seg_found = os.path.join(seg_mask_dir, m)
                break
        for m in dex_files:
            if os.path.splitext(m)[0] == base:
                dex_found = os.path.join(dex_mask_dir, m)
                break
        for m in lab_files:
            if os.path.splitext(m)[0] == base:
                lab_found = os.path.join(lab_mask_dir, m)
                break
        if seg_found is None:
            print(f"警告：在 {seg_mask_dir} 中找不到 {img} 对应的 Segmentation mask。")
        if dex_found is None:
            print(f"警告：在 {dex_mask_dir} 中找不到 {img} 对应的 Dexined mask。")
        if lab_found is None:
            print(f"警告：在 {lab_mask_dir} 中找不到 {img} 对应的 LAB mask。")
        if seg_found and dex_found and lab_found:
            pairs.append((os.path.join(image_dir, img), seg_found, dex_found, lab_found))
    return pairs

def apply_mixup_quad(img1, mask1, dex1, lab1, img2, mask2, dex2, lab2):
    target_h, target_w = img1.shape[:2]
    img2 = cv2.resize(img2, (target_w, target_h))
    mask2 = cv2.resize(mask2, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    dex2  = cv2.resize(dex2,  (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    lab2  = cv2.resize(lab2,  (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    if len(img1.shape) != len(img2.shape):
        if len(img1.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    alpha = random.uniform(0.3, 0.7)
    mixed_img  = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    mixed_mask = cv2.addWeighted(mask1, alpha, mask2, 1 - alpha, 0)
    mixed_dex  = cv2.addWeighted(dex1,  alpha, dex2,  1 - alpha, 0)
    mixed_lab  = cv2.addWeighted(lab1,  alpha, lab2,  1 - alpha, 0)
    return mixed_img, mixed_mask, mixed_dex, mixed_lab

def apply_mosaic_quad(image_list, mask_list, dex_list, lab_list, original_size=(512,512)):
    h, w = original_size
    mosaic_img = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    mosaic_mask = np.zeros((h*2, w*2), dtype=np.uint8)
    mosaic_dex  = np.zeros((h*2, w*2), dtype=np.uint8)
    mosaic_lab  = np.zeros((h*2, w*2), dtype=np.uint8)
    for i in range(4):
        img = image_list[i]
        mask = mask_list[i]
        dex  = dex_list[i]
        lab  = lab_list[i]
        scale = random.uniform(0.5, 1.5)
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h < h or new_w < w:
            scale = max(h / img.shape[0], w / img.shape[1])
            new_h, new_w = int(img.shape[0]*scale), int(img.shape[1]*scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        dex_resized  = cv2.resize(dex,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        lab_resized  = cv2.resize(lab,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        x_max = max(0, new_w - w)
        y_max = max(0, new_h - h)
        x = random.randint(0, x_max) if x_max > 0 else 0
        y = random.randint(0, y_max) if y_max > 0 else 0
        if new_h >= h and new_w >= w:
            img_crop = img_resized[y:y+h, x:x+w]
            mask_crop = mask_resized[y:y+h, x:x+w]
            dex_crop  = dex_resized[y:y+h, x:x+w]
            lab_crop  = lab_resized[y:y+h, x:x+w]
        else:
            img_crop = cv2.resize(img_resized, (w, h))
            mask_crop = cv2.resize(mask_resized, (w, h), interpolation=cv2.INTER_NEAREST)
            dex_crop  = cv2.resize(dex_resized, (w, h), interpolation=cv2.INTER_NEAREST)
            lab_crop  = cv2.resize(lab_resized, (w, h), interpolation=cv2.INTER_NEAREST)
        if i == 0:
            mosaic_img[:h, :w] = img_crop
            mosaic_mask[:h, :w] = mask_crop
            mosaic_dex[:h, :w]  = dex_crop
            mosaic_lab[:h, :w]  = lab_crop
        elif i == 1:
            mosaic_img[:h, w:] = img_crop
            mosaic_mask[:h, w:] = mask_crop
            mosaic_dex[:h, w:]  = dex_crop
            mosaic_lab[:h, w:]  = lab_crop
        elif i == 2:
            mosaic_img[h:, :w] = img_crop
            mosaic_mask[h:, :w] = mask_crop
            mosaic_dex[h:, :w]  = dex_crop
            mosaic_lab[h:, :w]  = lab_crop
        else:
            mosaic_img[h:, w:] = img_crop
            mosaic_mask[h:, w:] = mask_crop
            mosaic_dex[h:, w:]  = dex_crop
            mosaic_lab[h:, w:]  = lab_crop
    final_img  = cv2.resize(mosaic_img,  (w, h))
    final_mask = cv2.resize(mosaic_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    final_dex  = cv2.resize(mosaic_dex,  (w, h), interpolation=cv2.INTER_NEAREST)
    final_lab  = cv2.resize(mosaic_lab,  (w, h), interpolation=cv2.INTER_NEAREST)
    return final_img, final_mask, final_dex, final_lab

def augment_data_quad(pairs, output_image_dir, output_seg_dir, output_dex_dir, output_lab_dir, augmentations, augment_times=20):
    loaded_quads = []
    for img_path, seg_mask_path, dex_mask_path, lab_mask_path in pairs:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        seg = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
        dex = cv2.imread(dex_mask_path, cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(lab_mask_path, cv2.IMREAD_GRAYSCALE)
        loaded_quads.append((img, seg, dex, lab))
    all_images = [x[0] for x in loaded_quads]
    all_segs   = [x[1] for x in loaded_quads]
    all_dexs   = [x[2] for x in loaded_quads]
    all_labs   = [x[3] for x in loaded_quads]
    for idx in tqdm(range(len(loaded_quads)), desc="Augmenting"):
        img, seg, dex, lab = loaded_quads[idx]
        base_name = os.path.splitext(os.path.basename(pairs[idx][0]))[0]
        for i in range(augment_times):
            augmented = augmentations(
                image=img.copy(), 
                mask=seg.copy(), 
                dexined_mask=dex.copy(),
                lab_mask=lab.copy()
            )
            aug_img = augmented['image']
            aug_seg = augmented['mask']
            aug_dex = augmented['dexined_mask']
            aug_lab = augmented['lab_mask']
            # if random.random() < 0.3:
            #     other_idx = random.randint(0, len(loaded_quads)-1)
            #     o_img, o_seg, o_dex, o_lab = loaded_quads[other_idx]
            #     aug_img, aug_seg, aug_dex, aug_lab = apply_mixup_quad(
            #         aug_img, aug_seg, aug_dex, aug_lab,
            #         o_img.copy(), o_seg.copy(), o_dex.copy(), o_lab.copy()
            #     )
            # if random.random() < 0.2 and len(loaded_quads) >= 4:
            #     indices = [idx] + random.sample(range(len(loaded_quads)), 3)
            #     mosaic_imgs = [all_images[j] for j in indices]
            #     mosaic_segs = [all_segs[j] for j in indices]
            #     mosaic_dexs = [all_dexs[j] for j in indices]
            #     mosaic_labs = [all_labs[j] for j in indices]
            #     aug_img, aug_seg, aug_dex, aug_lab = apply_mosaic_quad(mosaic_imgs, mosaic_segs, mosaic_dexs, mosaic_labs)
            img_save_path = os.path.join(output_image_dir, f"{base_name}_aug_{i}.jpg")
            seg_save_path = os.path.join(output_seg_dir, f"{base_name}_aug_{i}.png")
            dex_save_path = os.path.join(output_dex_dir, f"{base_name}_aug_{i}.png")
            lab_save_path = os.path.join(output_lab_dir, f"{base_name}_aug_{i}.png")
            cv2.imwrite(img_save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(seg_save_path, aug_seg)
            cv2.imwrite(dex_save_path, aug_dex)
            cv2.imwrite(lab_save_path, aug_lab)

def main():
    # 设定随机种子，确保每次概率生成一致
    random.seed(SEED)
    np.random.seed(SEED)
    
    original_image_dir = f"{pparent_directory}/VOCdevkit_dilate/VOC2007/JPEGImages"
    original_seg_dir = os.path.join(os.path.dirname(original_image_dir), "SegmentationClass")
    dexined_dir = os.path.join(os.path.dirname(original_image_dir), "mask_output_folder")
    lab_dir = os.path.join(os.path.dirname(original_image_dir), "LAB_dir")
    
    augmented_image_dir = os.path.join(os.path.dirname(original_image_dir), "JPEGImages_Aug")
    augmented_seg_dir   = os.path.join(os.path.dirname(original_image_dir), "SegmentationClass_Aug")
    augmented_dex_dir   = os.path.join(os.path.dirname(original_image_dir), "Dexined_Aug")
    augmented_lab_dir   = os.path.join(os.path.dirname(original_image_dir), "LAB_Aug")
    
    create_output_dirs(augmented_image_dir, augmented_seg_dir, augmented_dex_dir, augmented_lab_dir)
    
    pairs = get_image_quadruple_pairs(original_image_dir, original_seg_dir, dexined_dir, lab_dir)
    print(f"共找到 {len(pairs)} 个 image + mask + dexined_mask + lab_mask 四元文件。")
    
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
            shear=(-15,15),
            translate_percent=(-0.1,0.1),
            scale=(0.8,1.2),
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3,0.3),
            contrast_limit=(-0.3,0.3),
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3,7), p=0.3),
        A.GaussNoise(var_limit=(10.0,50.0), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.RandomGamma(gamma_limit=(80,120), p=0.3),
        A.RandomSunFlare(src_radius=100, num_flare_circles_lower=3, num_flare_circles_upper=6, p=0.1),
        A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=0.1),
        A.PixelDropout(dropout_prob=0.01, p=0.1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3)
    ], additional_targets={'dexined_mask': 'mask', 'lab_mask': 'mask'})
    
    augment_data_quad(pairs, augmented_image_dir, augmented_seg_dir, augmented_dex_dir, augmented_lab_dir, augmentations, augment_times=10)
    print("数据增强完成！")

if __name__ == "__main__":
    main()