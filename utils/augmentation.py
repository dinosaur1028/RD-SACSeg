import os
import cv2
import random
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
            pairs.append((os.path.join(image_dir, img), os.path.join(mask_dir, msk)))
        else:
            print(f"Mismatch: {img} and {msk} do not match.")
    return pairs

def augment_data(pairs, output_image_dir, output_mask_dir, augmentations, augment_times=5):
    for img_path, mask_path in tqdm(pairs, desc="Augmenting"):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        for i in range(augment_times):
            augmented = augmentations(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']

            # 生成新的文件名
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
            new_img_name = f"{img_name}_aug_{i}.png"
            new_mask_name = f"{mask_name}_aug_{i}.png"

            # 保存增强后的图像和掩码
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_image_dir, new_img_name), aug_image_bgr)
            cv2.imwrite(os.path.join(output_mask_dir, new_mask_name), aug_mask)

def main():
    # 输入和输出目录
    original_image_dir = "/dataset/zhuluoji/unet/unet-pytorch-main/VOCdevkit/VOC2007/JPEGImages"  # 替换为你的原始图像文件夹路径
    original_mask_dir = "/dataset/zhuluoji/unet/unet-pytorch-main/VOCdevkit/VOC2007/SegmentationClass"    # 替换为你的原始掩码文件夹路径
    augmented_image_dir = "/dataset/zhuluoji/unet/unet-pytorch-main/VOCdevkit/VOC2007/aug_images"  # 替换为你想保存增强图像的文件夹路径
    augmented_mask_dir = "/dataset/zhuluoji/unet/unet-pytorch-main/VOCdevkit/VOC2007/aug_masks"    # 替换为你想保存增强掩码的文件夹路径

    create_output_dirs(augmented_image_dir, augmented_mask_dir)

    pairs = get_image_mask_pairs(original_image_dir, original_mask_dir)
    print(f"Found {len(pairs)} image-mask pairs.")

    # 定义数据增强管道
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.GridDistortion(p=0.2),
        A.HueSaturationValue(p=0.3),
        A.RandomScale(scale_limit=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    ], additional_targets={'mask': 'mask'})

    # 执行数据增强
    augment_data(pairs, augmented_image_dir, augmented_mask_dir, augmentations, augment_times=5)
    print("数据增强完成！")

if __name__ == "__main__":
    main()
