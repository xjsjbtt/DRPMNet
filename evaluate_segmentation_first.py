#!/usr/bin/env python3
"""
DINOv3图像分割评估和可视化 - 使用metrics.py计算指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import json
from tqdm import tqdm
import time

# 导入metrics.py中的函数
from mertics import ConfusionMatrix, get_score, get_score_sum

from datanew import LandslideDataset
from seg_dinov3_res18 import create_model

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows: 微软雅黑; Linux: 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


class SegmentationEvaluator:
    """分割模型评估器 - 使用metrics.py计算指标"""

    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

        # 评估结果
        self.results = {}

    def evaluate(self):
        """全面评估模型 - 使用metrics.py函数"""
        print("开始模型评估 ")

        self.model.eval()
        # 初始化混淆矩阵
        total_confusion_matrix = np.zeros((2, 2))

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="评估中"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                dems = batch['dem'].to(self.device)

                # 前向传播
                logits = self.model(images, dems)
                if logits.dim() == 3:
                    logits = logits.unsqueeze(1)
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()

                # 使用metrics.py中的函数计算混淆矩阵
                batch_cm = ConfusionMatrix(2, predictions, labels)
                total_confusion_matrix += batch_cm

        # 使用metrics.py函数计算指标
        precision, recall, f1score, iou, kc = get_score(total_confusion_matrix)
        cls_precision, cls_recall, cls_f1, cls_iou, kappa = get_score_sum(total_confusion_matrix)

        # 组织结果
        self.results = {
            'overall_accuracy': float(np.diag(total_confusion_matrix).sum() / (total_confusion_matrix.sum() + 1e-6)),
            'kappa': float(kappa),
            'class_0_precision': float(precision[0]),
            'class_0_recall': float(recall[0]),
            'class_0_f1': float(f1score[0]),
            'class_0_iou': float(iou[0]),
            'class_1_precision': float(precision[1]),
            'class_1_recall': float(recall[1]),
            'class_1_f1': float(f1score[1]),
            'class_1_iou': float(iou[1]),
            'confusion_matrix': total_confusion_matrix.tolist(),
            'landslide_focused': {
                'landslide_iou': float(iou[1]),
                'landslide_precision': float(precision[1]),
                'landslide_recall': float(recall[1]),
                'landslide_f1': float(f1score[1]),
                'true_positives': int(total_confusion_matrix[1, 1]),
                'false_positives': int(total_confusion_matrix[0, 1]),
                'false_negatives': int(total_confusion_matrix[1, 0])
            }
        }

        return self.results

    def print_results(self):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("模型评估结果 (使用metrics.py计算)")
        print("=" * 60)

        r = self.results
        print(f"整体准确率 (OA): {r['overall_accuracy']:.4f}")
        print(f"Kappa系数: {r['kappa']:.4f}")

        print(f"\n各类别指标:")
        print(
            f"  背景 - Precision: {r['class_0_precision']:.4f}, Recall: {r['class_0_recall']:.4f}, F1: {r['class_0_f1']:.4f}, IoU: {r['class_0_iou']:.4f}")
        print(
            f"  滑坡 - Precision: {r['class_1_precision']:.4f}, Recall: {r['class_1_recall']:.4f}, F1: {r['class_1_f1']:.4f}, IoU: {r['class_1_iou']:.4f}")

        # 滑坡专项评估结果
        lf = r['landslide_focused']
        print(f"\n{'=' * 60}")
        print("滑坡检测专项评估")
        print("=" * 60)
        print(f"滑坡IoU: {lf['landslide_iou']:.4f}")
        print(f"滑坡精确率: {lf['landslide_precision']:.4f}")
        print(f"滑坡召回率: {lf['landslide_recall']:.4f}")
        print(f"滑坡F1分数: {lf['landslide_f1']:.4f}")

        print(f"\n检测结果统计:")
        print(f"  真阳性 (正确检测): {lf['true_positives']} 像素")
        print(f"  假阳性 (误检): {lf['false_positives']} 像素")
        print(f"  假阴性 (漏检): {lf['false_negatives']} 像素")

        # 计算并显示漏检率和误检率
        cm = np.array(r['confusion_matrix'])
        total_landslide = lf['true_positives'] + lf['false_negatives']
        total_background = lf['false_positives'] + cm[0, 0]

        miss_rate = lf['false_negatives'] / total_landslide if total_landslide > 0 else 0
        false_alarm_rate = lf['false_positives'] / total_background if total_background > 0 else 0

        print(f"\n问题分析:")
        print(f"  漏检率: {miss_rate:.2%} ({lf['false_negatives']}/{total_landslide} 滑坡像素未被检测)")
        print(f"  误检率: {false_alarm_rate:.2%} ({lf['false_positives']}/{total_background} 背景像素被误判)")

        # 性能评估
        print(f"\n模型评估:")
        if lf['landslide_iou'] > 0.8:
            print("优秀: IoU > 0.8，滑坡检测效果优秀")
        elif lf['landslide_iou'] > 0.6:
            print("中等: IoU ≈ 0.6-0.8，滑坡检测效果中等，存在改进空间")
        elif lf['landslide_iou'] > 0.4:
            print("一般: IoU ≈ 0.4-0.6，滑坡检测效果一般，需要改进")
        else:
            print("较差: IoU < 0.4，滑坡检测效果较差")

        print(f"\n混淆矩阵:")
        print(f"  背景 -> 背景: {cm[0, 0]}")
        print(f"  背景 -> 滑坡: {cm[0, 1]}")
        print(f"  滑坡 -> 背景: {cm[1, 0]}")
        print(f"  滑坡 -> 滑坡: {cm[1, 1]}")

    def generate_report(self, save_path=None):
        """生成详细评估报告 - 使用metrics.py计算结果"""
        if not self.results:
            print("没有评估结果，请先运行evaluate()方法")
            return

        r = self.results
        lf = r['landslide_focused']
        cm = np.array(r['confusion_matrix'])

        report = []
        report.append("DINOv3图像分割模型评估报告 (使用metrics.py计算)")
        report.append("=" * 60)
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 整体性能指标
        report.append("整体性能指标:")
        report.append(f"  整体准确率 (OA): {r['overall_accuracy']:.4f}")
        report.append(f"  Kappa系数: {r['kappa']:.4f}")
        report.append("")

        # 各类别详细指标
        report.append("各类别详细指标:")
        report.append(
            f"  背景 - Precision: {r['class_0_precision']:.4f}, Recall: {r['class_0_recall']:.4f}, F1: {r['class_0_f1']:.4f}, IoU: {r['class_0_iou']:.4f}")
        report.append(
            f"  滑坡 - Precision: {r['class_1_precision']:.4f}, Recall: {r['class_1_recall']:.4f}, F1: {r['class_1_f1']:.4f}, IoU: {r['class_1_iou']:.4f}")
        report.append("")

        # 滑坡专项评估
        report.append("=" * 60)
        report.append("滑坡检测专项评估")
        report.append("=" * 60)
        report.append("滑坡检测性能指标:")
        report.append(f"  滑坡IoU: {lf['landslide_iou']:.4f}")
        report.append(f"  滑坡精确率: {lf['landslide_precision']:.4f}")
        report.append(f"  滑坡召回率: {lf['landslide_recall']:.4f}")
        report.append(f"  滑坡F1分数: {lf['landslide_f1']:.4f}")
        report.append("")

        report.append("检出统计:")
        report.append(f"  真阳性 (TP): {lf['true_positives']}")
        report.append(f"  假阳性 (FP): {lf['false_positives']}")
        report.append(f"  假阴性 (FN): {lf['false_negatives']}")
        report.append("")

        # 计算漏检率和误检率
        total_landslide = lf['true_positives'] + lf['false_negatives']
        total_background = lf['false_positives'] + cm[0, 0]
        miss_rate = lf['false_negatives'] / total_landslide if total_landslide > 0 else 0
        false_alarm_rate = lf['false_positives'] / total_background if total_background > 0 else 0

        report.append("问题分析:")
        report.append(f"  漏检率: {miss_rate:.2%}")
        report.append(f"  误检率: {false_alarm_rate:.2%}")
        report.append("")

        # 模型评估结论
        report.append("模型评估结论:")
        if lf['landslide_iou'] > 0.8:
            report.append("优秀: 滑坡IoU > 0.8")
        elif lf['landslide_iou'] > 0.6:
            report.append("中等: 滑坡IoU ≈ 0.6-0.8")
        elif lf['landslide_iou'] > 0.4:
            report.append("一般: 滑坡IoU ≈ 0.4-0.6")
        else:
            report.append("较差: 滑坡IoU < 0.4")

        # 混淆矩阵信息
        report.append("")
        report.append("混淆矩阵:")
        report.append(f"  背景 -> 背景: {cm[0, 0]}")
        report.append(f"  背景 -> 滑坡: {cm[0, 1]}")
        report.append(f"  滑坡 -> 背景: {cm[1, 0]}")
        report.append(f"  滑坡 -> 滑坡: {cm[1, 1]}")

        # 保存报告
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            print(f"详细评估报告已保存到: {save_path}")

        return '\n'.join(report)

    def save_results(self, save_path=None):
        """保存评估结果到JSON文件"""
        if save_path is None:
            save_path = './outputs/evaluation_results.json'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"评估结果已保存到: {save_path}")


class SegmentationVisualizer:
    """分割结果可视化器"""

    def __init__(self, model, test_dataset, device='cuda'):
        self.model = model.to(device)
        self.test_dataset = test_dataset
        self.device = device

    def visualize_predictions(self, num_samples=6, save_path=None):
        """可视化预测结果"""
        self.model.eval()

        # 随机选择样本
        indices = np.random.choice(len(self.test_dataset), num_samples, replace=False)

        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        with torch.no_grad():
            for i, idx in enumerate(indices):
                # 获取数据
                sample = self.test_dataset[idx]
                image = sample['image'].unsqueeze(0).to(self.device)
                label = sample['label']
                dem = sample['dem']
                filename = sample['filename']

                # 预测
                logits = self.model(image, dem.unsqueeze(0).to(self.device))
                if logits.dim() == 3:
                    logits = logits.unsqueeze(1)
                probabilities = torch.sigmoid(logits)
                prediction = (probabilities > 0.5).long().squeeze(0).cpu()

                # 转换为numpy用于显示
                image_np = self.denormalize_image(image.squeeze(0).cpu())
                label_np = label.numpy()
                dem_np = dem.squeeze(0).numpy()
                pred_np = prediction.numpy()

                # 修复形状问题：确保 pred_np 是 2D 数组
                if pred_np.shape[0] == 1:
                    pred_np = pred_np[0]  # (1,H,W) -> (H,W)
                else:
                    pred_np = pred_np[i]

                    # 确保 label_np 也是 2D 数组
                if label_np.ndim == 3:
                    label_np = label_np.squeeze(0)

                # 显示原始图像
                axes[i, 0].imshow(image_np)
                axes[i, 0].set_title(f'原始图像\n{filename}')
                axes[i, 0].axis('off')

                # 显示DEM
                axes[i, 1].imshow(dem_np, cmap='terrain')
                axes[i, 1].set_title('DEM')
                axes[i, 1].axis('off')

                # 显示真实标签
                axes[i, 2].imshow(label_np, cmap='viridis')
                axes[i, 2].set_title('label')
                axes[i, 2].axis('off')

                # 显示预测结果
                axes[i, 3].imshow(pred_np, cmap='viridis')
                axes[i, 3].set_title('pre')
                axes[i, 3].axis('off')

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测结果已保存到: {save_path}")
        plt.show()

    def denormalize_image(self, tensor):
        """反归一化图像用于显示"""
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).numpy()

    def create_confusion_matrix_plot(self, confusion_matrix, save_path=None):
        """创建混淆矩阵热力图"""
        plt.figure(figsize=(8, 6))

        # 计算百分比
        cm_percent = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100

        # 创建热力图
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=['背景', '滑坡'],
                    yticklabels=['背景', '滑坡'])

        plt.title('混淆矩阵 (百分比)')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        plt.show()

    def create_probability_heatmap(self, image, dem, save_path=None):
        """创建概率热力图"""
        self.model.eval()

        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(self.device)
            dem_tensor = dem.unsqueeze(0).to(self.device)

            logits = self.model(image_tensor, dem_tensor)
            if logits.dim() == 3:
                logits = logits.unsqueeze(1)
            probabilities = torch.sigmoid(logits)

            # 获取滑坡概率
            landslide_prob = probabilities[0, 0].cpu().numpy()  # 注意：这里可能需要调整索引

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        image_np = self.denormalize_image(image)
        axes[0].imshow(image_np)
        axes[0].set_title('label')
        axes[0].axis('off')

        # DEM
        axes[1].imshow(dem.squeeze(0).numpy(), cmap='terrain')
        axes[1].set_title('DEM')
        axes[1].axis('off')

        # 概率热力图
        im = axes[2].imshow(landslide_prob, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('heat')
        axes[2].axis('off')

        # 添加颜色条
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"概率热力图已保存到: {save_path}")
        plt.show()
        plt.show()

    def overlay_pred(self, img, pred):
        """
        img: RGB uint8
        pred: [H,W] 概率图 (0~1)
        """

        # -------- 二值化 --------
        mask = (pred > 0.5).astype(np.uint8)

        overlay = img.copy().astype(np.float32)

        # -------- 取得 JET colormap 的深蓝色 --------
        jet_zero = cv2.applyColorMap(
            np.array([[0]], dtype=np.uint8),
            cv2.COLORMAP_JET
        )[0, 0]  # BGR

        blue_rgb = jet_zero[::-1].astype(np.float32)

        # -------- 背景区域变暗 + 蓝调 --------
        alpha_bg = 0.25  # 控制蓝色强度

        overlay[mask == 0] = (
                overlay[mask == 0] * (1 - alpha_bg)
                + blue_rgb * alpha_bg
        )

        # -------- 前景棕色填充 --------
        brown = np.array([160, 80, 45], dtype=np.float32)
        alpha_fg = 0.65

        overlay[mask == 1] = (
                overlay[mask == 1] * (1 - alpha_fg)
                + brown * alpha_fg
        )

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # -------- 青色细轮廓 --------
        contours, _ = cv2.findContours(
            mask * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            (0, 255, 255),  # 青色
            thickness=1
        )

        return overlay

    def export_per_sample_results(self, output_root='./outputs/res18_image_results'):

        self.model.eval()
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for idx in tqdm(range(len(self.test_dataset)), desc="Exporting"):

                sample = self.test_dataset[idx]

                image = sample['image'].unsqueeze(0).to(self.device)
                dem = sample['dem'].unsqueeze(0).to(self.device)
                label = sample['label']
                filename = Path(sample['filename']).stem

                # ---------- 前向传播 ----------
                logits = self.model(image, dem)

                if logits.dim() == 3:
                    logits = logits.unsqueeze(1)

                # 兼容 2 类输出
                if logits.shape[1] == 2:
                    probs = torch.softmax(logits, dim=1)
                    pred_prob = probs[0, 1].cpu().numpy()
                else:
                    pred_prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

                # ---------- 创建文件夹 ----------
                sample_dir = output_root / filename
                sample_dir.mkdir(parents=True, exist_ok=True)

                # ---------- 保存 RGB ----------
                rgb = self.denormalize_image(image.squeeze(0).cpu())
                rgb_uint8 = (rgb * 255).astype(np.uint8)

                cv2.imwrite(
                    str(sample_dir / "rgb.png"),
                    cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
                )

                # ---------- 保存 DEM ----------
                dem_np = dem.squeeze().cpu().numpy()
                dem_norm = cv2.normalize(dem_np, None, 0, 255, cv2.NORM_MINMAX)

                cv2.imwrite(
                    str(sample_dir / "dem.png"),
                    dem_norm.astype(np.uint8)
                )

                # ---------- 保存 label ----------
                label_np = label.squeeze().numpy()
                label_img = (label_np * 255).astype(np.uint8)

                cv2.imwrite(
                    str(sample_dir / "label.png"),
                    label_img
                )

                # ---------- 生成叠加图 ----------
                overlay = self.overlay_pred(rgb_uint8, pred_prob)

                cv2.imwrite(
                    str(sample_dir / "pred_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                )

        print(f"\nSaved to: {output_root}")

