import os
import random
import shutil
from pathlib import Path

# ===================== 配置 =====================
INPUT_DIR = Path("newdata")          # 原始数据目录
OUTPUT_DIR = Path("newdata_split")   # 输出目录
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# ===================== 固定随机种子 =====================
random.seed(SEED)

# ===================== 按数字ID匹配所有文件 =====================
def get_id(fname):
    """从文件名中提取数字部分，比如 image_3941.tif → 3941"""
    return fname.split("_")[-1].split(".")[0]

# 1. 读取所有 image 文件，并记录 ID
img_dir = INPUT_DIR / "image"
img_files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
img_id_map = {get_id(f): f for f in img_files}
print(f"image 文件夹找到 {len(img_files)} 个文件")

# 2. 读取所有 dem 文件，并记录 ID
dem_dir = INPUT_DIR / "dem"
dem_files = [f for f in os.listdir(dem_dir) if f.endswith(".tif")]
dem_id_map = {get_id(f): f for f in dem_files}
print(f"dem 文件夹找到 {len(dem_files)} 个文件")

# 3. 读取所有 label 文件，并记录 ID
label_dir = INPUT_DIR / "label"
label_files = [f for f in os.listdir(label_dir) if f.endswith(".tif")]
label_id_map = {get_id(f): f for f in label_files}
print(f"label 文件夹找到 {len(label_files)} 个文件")

# 4. 取三者的交集（只保留三者都有的 ID）
common_ids = sorted(list(set(img_id_map.keys()) & set(dem_id_map.keys()) & set(label_id_map.keys())))
random.shuffle(common_ids)
print(f"三者都存在的样本数: {len(common_ids)}")

# ===================== 划分 =====================
total = len(common_ids)
train_num = int(total * TRAIN_RATIO)
val_num = int(total * VAL_RATIO)

train_ids = common_ids[:train_num]
val_ids = common_ids[train_num : train_num + val_num]
test_ids = common_ids[train_num + val_num:]

print(f"训练集: {len(train_ids)}")
print(f"验证集: {len(val_ids)}")
print(f"测试集: {len(test_ids)}")

# ===================== 创建目录 =====================
for split in ["train", "val", "test"]:
    for folder in ["image", "dem", "label"]:
        os.makedirs(OUTPUT_DIR / split / folder, exist_ok=True)

# ===================== 复制文件（按ID匹配，自动处理前缀不同） =====================
def copy_files(ids, split):
    for file_id in ids:
        # 复制 image
        img_fname = img_id_map[file_id]
        src_img = img_dir / img_fname
        dst_img = OUTPUT_DIR / split / "image" / img_fname
        shutil.copy(src_img, dst_img)

        # 复制 dem
        dem_fname = dem_id_map[file_id]
        src_dem = dem_dir / dem_fname
        dst_dem = OUTPUT_DIR / split / "dem" / dem_fname
        shutil.copy(src_dem, dst_dem)

        # 复制 label
        label_fname = label_id_map[file_id]
        src_lbl = label_dir / label_fname
        dst_lbl = OUTPUT_DIR / split / "label" / label_fname
        shutil.copy(src_lbl, dst_lbl)

copy_files(train_ids, "train")
copy_files(val_ids, "val")
copy_files(test_ids, "test")

print(" 输出目录：newdata_split")