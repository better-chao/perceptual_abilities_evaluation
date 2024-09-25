import os
import shutil

def _read_pairs(txt_file):
    """
    读取 camvid_train.txt, camvid_val.txt 等文件，返回图像和标签路径配对列表
    """
    pairs = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img, label = line.strip().split()
            pairs.append((img, label))
    return pairs

def copy_files(pairs, img_src_dir, label_src_dir, img_dst_dir, label_dst_dir):
    """
    根据配对列表将图像和标签文件复制到目标目录
    """
    if not os.path.exists(img_dst_dir):
        os.makedirs(img_dst_dir)
    if not os.path.exists(label_dst_dir):
        os.makedirs(label_dst_dir)

    for img_path, label_path in pairs:
        # 获取文件的完整路径
        img_full_path = os.path.join(img_src_dir, os.path.basename(img_path))
        label_full_path = os.path.join(label_src_dir, os.path.basename(label_path))

        # 复制图像和标签到目标文件夹
        shutil.copy(img_full_path, img_dst_dir)
        shutil.copy(label_full_path, label_dst_dir)

        print(f"Copied: {img_full_path} -> {img_dst_dir}")
        print(f"Copied: {label_full_path} -> {label_dst_dir}")

def organize_dataset(splits_dir, img_src_dir, label_src_dir, save_dir):
    """
    读取已有的txt文件，并将图像和标签划分到train、val、trainval和test子文件夹中
    """
    # 定义四个分割文件
    split_files = ['camvid_train.txt', 'camvid_val.txt', 'camvid_trainval.txt', 'camvid_test.txt']

    for split_file in split_files:
        split_path = os.path.join(splits_dir, split_file)
        pairs = _read_pairs(split_path)

        # 根据文件名决定存放的子文件夹
        split_name = split_file.split('.')[0].split('_')[-1]

        img_dst_dir = os.path.join(save_dir, split_name, 'images')
        label_dst_dir = os.path.join(save_dir, split_name, 'labels')

        # 复制图像和标签文件到目标文件夹
        copy_files(pairs, img_src_dir, label_src_dir, img_dst_dir, label_dst_dir)

if __name__ == '__main__':
    # 定义源图像和标签路径，保存路径以及分割txt文件路径
    img_src_dir = '/gpfsdata/home/wanqiao/dataset/CamVid/CamVid_RGB'
    label_src_dir = '/gpfsdata/home/wanqiao/dataset/CamVid/CamVidGray'
    splits_dir = '/gpfsdata/home/wanqiao/dataset/CamVid/splits'
    save_dir = '/gpfsdata/home/wanqiao/dataset/CamVid/sorted_camvid'

    # 执行数据集划分整理
    organize_dataset(splits_dir, img_src_dir, label_src_dir, save_dir)
