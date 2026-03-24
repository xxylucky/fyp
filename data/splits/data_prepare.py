import os
import cv2
import numpy as np
import glob
import shutil


def preprocess_busi_dataset(raw_dir, output_dir):
    """
    处理原始 BUSI 数据集: 合并多掩码, 仅保留 benign 和 malignant, 并分离原图与标签。
    """
    # 论文中明确指出只使用 benign 和 malignant
    target_classes = ['benign', 'malignant']

    # 创建目标文件夹 (模拟代码需要的 leftImg 和 gtFine 结构)
    leftImg_dir = os.path.join(output_dir, 'leftImg', 'all_data')
    gtFine_dir = os.path.join(output_dir, 'gtFine', 'all_data')
    os.makedirs(leftImg_dir, exist_ok=True)
    os.makedirs(gtFine_dir, exist_ok=True)

    processed_count = 0

    for cls in target_classes:
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"警告: 未找到文件夹 {cls_dir}")
            continue

        # 遍历该类别下所有的原图（不包含 '_mask' 的 .png 文件）
        all_files = os.listdir(cls_dir)
        original_images = [f for f in all_files if '_mask' not in f and f.endswith('.png')]

        for img_name in original_images:
            # 1. 拷贝原图
            src_img_path = os.path.join(cls_dir, img_name)
            dst_img_path = os.path.join(leftImg_dir, img_name)
            shutil.copy(src_img_path, dst_img_path)

            # 2. 查找对应的所有掩码文件
            # 例如: benign (4).png -> 查找 benign (4)_mask*.png
            base_name = img_name.replace('.png', '')
            mask_pattern = os.path.join(cls_dir, f"{base_name}_mask*.png")
            mask_files = glob.glob(mask_pattern)

            if not mask_files:
                print(f"警告: {img_name} 没有找到对应的掩码文件！")
                continue

            # 3. 合并多个掩码
            combined_mask = None
            for mask_path in mask_files:
                # 以灰度图模式读取掩码
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if combined_mask is None:
                    combined_mask = mask_img
                else:
                    # 将多个掩码按位或（融合所有病灶区域）
                    combined_mask = np.maximum(combined_mask, mask_img)

            # 确保掩码是纯二值化图像 (0 和 255)
            _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

            # 4. 保存合并后的掩码
            # 注意：保存的掩码名字需要和原图一致，以匹配 dataset 中的 replace('leftImg', 'gtFine') 逻辑
            dst_mask_path = os.path.join(gtFine_dir, img_name)
            cv2.imwrite(dst_mask_path, combined_mask)

            processed_count += 1

    print(f"处理完成！共处理了 {processed_count} 张图像及其掩码。")
    print(f"原图存放于: {leftImg_dir}")
    print(f"掩码存放于: {gtFine_dir}")


if __name__ == '__main__':
    # 请将下面两个路径替换为你电脑上的实际路径
    RAW_DATA_DIRECTORY = './Dataset_BUSI_with_GT'  # 你下载解压后的原始文件夹
    OUTPUT_DIRECTORY = '../BUSI'  # 预处理后存放的目录

    preprocess_busi_dataset(RAW_DATA_DIRECTORY, OUTPUT_DIRECTORY)