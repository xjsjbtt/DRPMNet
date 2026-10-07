# DRPMNet
DRPMNet is a multimodal landslide detection model combining DINOv3 and ResNet18.

## Model
The DINOv3 model is directly incorporated. All subsequent parameter adjustments can be performed via `run_seg.py` located in the root directory.

## Datasets
- **newdata**: A private dataset. The provided segmentation code supports segmentation on this dataset and allows customization of segmentation pipelines as needed.
- **Bijie dataset**: Download from https://gpcv.whu.edu.cn/data/Bijie_pages.html. We adopt the landslide-containing subset of this dataset. Data augmentation methods are implemented in the dataloader file.

## DINOv3 Source
https://github.com/facebookresearch/dinov3
