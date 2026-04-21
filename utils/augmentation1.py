
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
    for img, msk in zip(image_files, mask_files):
        if os.path.splitext(img)[0] == os.path.splitext(msk)[0]:
            pairs.append((
                os.path.join(image_dir, img),
                os.path.join(mask_dir, msk)
            ))
        else:
            print(f"Mismatch: {img} and {msk} do not match.")
    return pairs

def apply_mixup(img1, mask1, img2, mask2):
    """MixUp数据增强实现（自动统一尺寸）"""
    # 统一目标尺寸（以第一张图片的尺寸为准）
    target_h, target_w = img1.shape[:2]
    
    # 调整第二张图片尺寸
    img2 = cv2.resize(img2, (target_w, target_h))
    mask2 = cv2.resize(mask2, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    # 统一通道数（如果原始图片是灰度图）
    if len(img1.shape) != len(img2.shape):
        if len(img1.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    
    alpha = random.uniform(0.3, 0.7)
    mixed_img = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
    mixed_mask = cv2.addWeighted(mask1, alpha, mask2, 1-alpha, 0)
    return mixed_img, mixed_mask


def apply_mosaic(images, masks, original_size=(512, 512)):
    """安全版Mosaic实现"""
    h, w = original_size  # 原始图片尺寸
    mosaic_img = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    mosaic_mask = np.zeros((h*2, w*2), dtype=np.uint8)

    for i in range(4):
        img = images[i]
        mask = masks[i]
        
        # 动态计算安全缩放比例
        scale = random.uniform(0.5, 1.5)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 保证缩放后尺寸不小于原始尺寸
        if new_h < h or new_w < w:
            scale = max(h/img.shape[0], w/img.shape[1])
            new_h, new_w = int(img.shape[0]*scale), int(img.shape[1]*scale)
        
        img = cv2.resize(img, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 计算安全裁剪范围
        x_max = max(0, new_w - w)
        y_max = max(0, new_h - h)
        x = random.randint(0, x_max) if x_max > 0 else 0
        y = random.randint(0, y_max) if y_max > 0 else 0
        
        # 执行裁剪
        img_crop = img[y:y+h, x:x+w] if (new_h >= h and new_w >= w) else cv2.resize(img, (w, h))
        mask_crop = mask[y:y+h, x:x+w] if (new_h >= h and new_w >= w) else cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 填充到mosaic画布
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

    # 最终调整到原始尺寸
    return (
        cv2.resize(mosaic_img, (w, h)),
        cv2.resize(mosaic_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    )


def augment_data(pairs, output_image_dir, output_mask_dir, augmentations, augment_times=20):
    # 预加载所有图片和mask
    loaded_pairs = []
    for img_path, mask_path in pairs:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        loaded_pairs.append((img, mask))
    
    all_images = [p[0] for p in loaded_pairs]
    all_masks = [p[1] for p in loaded_pairs]

    for idx in tqdm(range(len(loaded_pairs)), desc="Augmenting"):
        img, mask = loaded_pairs[idx]
        
        for i in range(augment_times):
            # Albumentations基础增强
            augmented = augmentations(image=img.copy(), mask=mask.copy())
            aug_img = augmented['image']
            aug_mask = augmented['mask']

            # 随机应用MixUp（30%概率）
            if random.random() < 0.3:
                other_idx = random.randint(0, len(loaded_pairs)-1)
                other_img = all_images[other_idx]
                other_mask = all_masks[other_idx]
                aug_img, aug_mask = apply_mixup(aug_img, aug_mask, 
                                              other_img, other_mask)

            # 随机应用Mosaic（20%概率）
            if random.random() < 0.2 and len(loaded_pairs) >= 4:
                indices = [idx] + random.sample(range(len(loaded_pairs)), 3)
                mosaic_imgs = [all_images[i] for i in indices]
                mosaic_masks = [all_masks[i] for i in indices]
                aug_img, aug_mask = apply_mosaic(mosaic_imgs, mosaic_masks)

            # 保存结果
            base_name = f"{os.path.splitext(os.path.basename(pairs[idx][0]))[0]}_aug_{i}"
            cv2.imwrite(os.path.join(output_image_dir, f"{base_name}.png"),
                       cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}.png"), aug_mask)

def main():
    original_image_dir = "/dataset/zhuluoji/unet/unet-pytorch-main/VOCdevkit/VOC2007/JPEGImages_O"  # 替换为你的原始图像文件夹路径
    original_mask_dir = "/dataset/zhuluoji/unet/unet-pytorch-main/VOCdevkit/VOC2007/SegmentationClass_O"    # 替换为你的原始掩码文件夹路径
    augmented_image_dir = "/dataset/zhuluoji/unet/unet-pytorch-main/VOCdevkit/VOC2007/JPEGImages"  # 替换为你想保存增强图像的文件夹路径
    augmented_mask_dir = "/dataset/zhuluoji/unet/unet-pytorch-main/VOCdevkit/VOC2007/SegmentationClass"    # 替换为你想保存增强掩码的文件夹路径

    create_output_dirs(augmented_image_dir, augmented_mask_dir)
    pairs = get_image_mask_pairs(original_image_dir, original_mask_dir)
    print(f"Found {len(pairs)} image-mask pairs.")

    # 增强管道（新增多种增强方式）
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

    augment_data(pairs, augmented_image_dir, augmented_mask_dir, augmentations, 10)
    print("数据增强完成！")

if __name__ == "__main__":
    main()
