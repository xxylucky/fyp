import os
import shutil


def organize_dataset_by_splits(splits_dir, all_leftImg_dir, all_gtFine_dir, output_base_dir):
    """
    根据给定的 .txt 划分文件，将平铺的图片自动分发到 train/val 物理目录中。
    """
    # 收集所有的 .txt 文件路径 (包括子目录下的 labeled/unlabeled/val)
    txt_files = []
    for root, _, files in os.walk(splits_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))

    if not txt_files:
        print(f"在 {splits_dir} 中没有找到任何 .txt 文件！")
        return

    processed_images = set()  # 用于记录已经复制过的文件，避免不同 txt 重复复制

    for txt_path in txt_files:
        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()

        for relative_path in lines:
            if not relative_path:
                continue

            # 相对路径例子: "leftImg/train/benign (70).png"
            # 提取文件名: "benign (70).png"
            file_name = os.path.basename(relative_path)

            # 如果这个文件已经处理过了，跳过
            if file_name in processed_images:
                continue

            # --- 处理原图 (leftImg) ---
            src_img = os.path.join(all_leftImg_dir, file_name)
            # 目标路径: output_base_dir/leftImg/train/benign (70).png
            dst_img = os.path.join(output_base_dir, relative_path)

            # --- 处理掩码图 (gtFine) ---
            src_mask = os.path.join(all_gtFine_dir, file_name)
            # 对应的掩码相对路径: "gtFine/train/benign (70).png"
            mask_relative_path = relative_path.replace('leftImg', 'gtFine')
            dst_mask = os.path.join(output_base_dir, mask_relative_path)

            # 如果原始文件存在，则创建目标文件夹并复制
            if os.path.exists(src_img) and os.path.exists(src_mask):
                os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                os.makedirs(os.path.dirname(dst_mask), exist_ok=True)

                shutil.copy(src_img, dst_img)
                shutil.copy(src_mask, dst_mask)
                processed_images.add(file_name)
            else:
                print(f"警告: 找不到源文件 {src_img} 或其掩码，跳过。")

    print(f"目录分组构建完成！共成功分配了 {len(processed_images)} 张图像及掩码。")
    print(f"输出目录已按照代码要求构建在: {output_base_dir}")


if __name__ == '__main__':
    # 请根据你实际存放的路径修改以下变量
    # 1. 存放作者给的各类 .txt 文件的根目录 (例如 72, 144, val.txt 所在的顶层)
    SPLITS_DIR = './BUSI'

    # 2. 上一步跑完预处理脚本后生成的全量图片存放地
    ALL_LEFTIMG_DIR = '../BUSI/leftImg/all_data'
    ALL_GTFINE_DIR = '../BUSI/gtFine/all_data'

    # 3. 最终要输出的、可以直接被 main.py 读取的根目录
    OUTPUT_BASE_DIR = '../BUSI'

    organize_dataset_by_splits(SPLITS_DIR, ALL_LEFTIMG_DIR, ALL_GTFINE_DIR, OUTPUT_BASE_DIR)