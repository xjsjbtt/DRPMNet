"""
DINOv3图像分割训练脚本 —— 使用metrics.py计算指标 (二分类)
改进：
- 使用metrics.py中的混淆矩阵计算方法
- 计算所有像素的指标（包括背景和前景）
- 添加Kappa系数、整体准确率等指标
- 支持 BCEWithLogitsLoss 和 CrossEntropyLoss 两种损失
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入metrics.py中的函数
from mertics import ConfusionMatrix, get_score, get_score_sum

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows: 微软雅黑; Linux: 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


class SegmentationTrainer:
    """分割模型训练器（二分类）- 使用metrics.py计算指标"""

    def __init__(self, model, train_loader, val_loader, device, lr, epoch):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_bce = True

        # 损失函数
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_ce = nn.CrossEntropyLoss()  # CrossEntropy 只支持 label [0,n_classes-1]

        # 优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch)

        # TensorBoard
        self.writer = SummaryWriter('runs/segmentation_experiment')

        # 训练历史
        self.train_history = {
            'train_loss': [], 'train_oa': [], 'train_pa': [], 'train_recall': [],
            'train_f1': [], 'train_iou': [], 'train_kappa': [],
            'val_loss': [], 'val_oa': [], 'val_pa': [], 'val_recall': [],
            'val_f1': [], 'val_iou': [], 'val_kappa': []
        }

        self.best_val_f1 = 0.0
        self.best_model_path = './outputs_train/model_adam_one_cross_data2_bce_c3.pth'
        Path(self.best_model_path).parent.mkdir(parents=True, exist_ok=True)
        Path('./outputs_train').mkdir(parents=True, exist_ok=True)

    # -------------------- label utils --------------------
    def _ensure_label_encoding(self, labels):
        """
        将 label 映射为 0/1，BCE 和 CrossEntropy 通用
        """
        # 转为 long
        if not labels.dtype.is_floating_point:
            labels = labels.to(torch.long)
        # 将 255 映射为 1
        labels = torch.where(labels == 255, torch.tensor(1, device=labels.device), labels)
        # 确保只有 0/1
        labels = torch.clamp(labels, 0, 1)
        return labels  # long, [B,H,W]

    # -------------------- epoch routines --------------------
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        confusion_matrix = np.zeros((2, 2))  # 二分类

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1} [Train]')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            dems = batch.get('dem', None)
            if dems is not None:
                dems = dems.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images, dems)

            labels = self._ensure_label_encoding(labels)

            if self.use_bce:
                labels_for_loss = labels.unsqueeze(1).float()  # [B,1,H,W]
                loss = self.criterion_bce(logits, labels_for_loss)
                predictions = (torch.sigmoid(logits) > 0.5).float().squeeze(1)  # [B,H,W]
            else:
                loss = self.criterion_ce(logits, labels)
                predictions = torch.argmax(logits, dim=1).float()

            loss.backward()
            self.optimizer.step()

            # metrics
            with torch.no_grad():
                batch_cm = ConfusionMatrix(2, predictions, labels)
                confusion_matrix += batch_cm
                precision, recall, f1score, iou, kc = get_score(batch_cm)
                precision_1 = precision[1]
                recall_1 = recall[1]
                f1_1 = f1score[1]
                iou_1 = iou[1]
                oa = np.diag(batch_cm).sum() / (batch_cm.sum() + 1e-6)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'OA': f'{oa:.4f}',
                'PA': f'{precision_1:.4f}',
                'Recall': f'{recall_1:.4f}',
                'F1': f'{f1_1:.4f}',
                'IoU': f'{iou_1:.4f}'
            })

        avg_loss = total_loss / max(1, num_batches)
        precision, recall, f1score, iou, kc = get_score(confusion_matrix)
        cls_precision, cls_recall, cls_f1, cls_iou, kappa = get_score_sum(confusion_matrix)
        precision_1 = precision[1]
        recall_1 = recall[1]
        f1_1 = f1score[1]
        iou_1 = iou[1]
        oa = np.diag(confusion_matrix).sum() / (confusion_matrix.sum() + 1e-6)
        print(f"Epoch {epoch + 1} Train Confusion Matrix:\n{confusion_matrix}")
        print(
            f"OA: {oa:.4f}, PA: {precision_1:.4f}, Recall: {recall_1:.4f}, F1: {f1_1:.4f}, IoU: {iou_1:.4f}, Kappa: {kappa:.4f}")
        return avg_loss, oa, precision_1, recall_1, f1_1, iou_1, kappa

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        confusion_matrix = np.zeros((2, 2))

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch + 1} [Val]')
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                dems = batch.get('dem', None)
                if dems is not None:
                    dems = dems.to(self.device)
                logits = self.model(images, dems)
                labels = self._ensure_label_encoding(labels)

                if self.use_bce:
                    labels_for_loss = labels.unsqueeze(1).float()
                    loss = self.criterion_bce(logits, labels_for_loss)

                    predictions = (torch.sigmoid(logits) > 0.5).float().squeeze(1)
                else:
                    loss = self.criterion_ce(logits, labels)
                    predictions = torch.argmax(logits, dim=1).float()

                batch_cm = ConfusionMatrix(2, predictions, labels)
                confusion_matrix += batch_cm
                precision, recall, f1score, iou, kc = get_score(batch_cm)
                precision_1 = precision[1]
                recall_1 = recall[1]
                f1_1 = f1score[1]
                iou_1 = iou[1]
                oa = np.diag(batch_cm).sum() / (batch_cm.sum() + 1e-6)

                total_loss += loss.item()
                num_batches += 1
                print("Loss:", loss.item())
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'OA': f'{oa:.4f}',
                    'PA': f'{precision_1:.4f}',
                    'Recall': f'{recall_1:.4f}',
                    'F1': f'{f1_1:.4f}',
                    'IoU': f'{iou_1:.4f}'
                })

        avg_loss = total_loss / max(1, num_batches)
        precision, recall, f1score, iou, kc = get_score(confusion_matrix)
        cls_precision, cls_recall, cls_f1, cls_iou, kappa = get_score_sum(confusion_matrix)
        precision_1 = precision[1]
        recall_1 = recall[1]
        f1_1 = f1score[1]
        iou_1 = iou[1]
        oa = np.diag(confusion_matrix).sum() / (confusion_matrix.sum() + 1e-6)
        print(f"Validation Epoch {epoch + 1} Confusion Matrix:\n{confusion_matrix}")
        print(
            f"OA: {oa:.4f}, PA: {precision_1:.4f}, Recall: {recall_1:.4f}, F1: {f1_1:.4f}, IoU: {iou_1:.4f}, Kappa: {kappa:.4f}")
        return avg_loss, oa, precision_1, recall_1, f1_1, iou_1, kappa

    # -------------------- training loop --------------------
    def train(self, num_epochs):
        print("============================================================")
        print("开始训练阶段 - 使用metrics.py计算指标")
        print("============================================================")
        print(f"开始训练，共{num_epochs}个epoch")
        print(f"设备: {self.device}")
        print(f"训练集批次数: {len(self.train_loader)}")
        print(f"验证集批次数: {len(self.val_loader)}")

        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_loss, train_oa, train_pa, train_recall, train_f1, train_iou, train_kappa = self.train_epoch(epoch)
            val_loss, val_oa, val_pa, val_recall, val_f1, val_iou, val_kappa = self.validate(epoch)

            self.scheduler.step()

            # 更新历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_oa'].append(train_oa)
            self.train_history['train_pa'].append(train_pa)
            self.train_history['train_recall'].append(train_recall)
            self.train_history['train_f1'].append(train_f1)
            self.train_history['train_iou'].append(train_iou)
            self.train_history['train_kappa'].append(train_kappa)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_oa'].append(val_oa)
            self.train_history['val_pa'].append(val_pa)
            self.train_history['val_recall'].append(val_recall)
            self.train_history['val_f1'].append(val_f1)
            self.train_history['val_iou'].append(val_iou)
            self.train_history['val_kappa'].append(val_kappa)

            # tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('OA/Train', train_oa, epoch)
            self.writer.add_scalar('OA/Val', val_oa, epoch)
            self.writer.add_scalar('PA/Train', train_pa, epoch)
            self.writer.add_scalar('PA/Val', val_pa, epoch)
            self.writer.add_scalar('Recall/Train', train_recall, epoch)
            self.writer.add_scalar('Recall/Val', val_recall, epoch)
            self.writer.add_scalar('F1/Train', train_f1, epoch)
            self.writer.add_scalar('F1/Val', val_f1, epoch)
            self.writer.add_scalar('IoU/Train', train_iou, epoch)
            self.writer.add_scalar('IoU/Val', val_iou, epoch)
            self.writer.add_scalar('Kappa/Train', train_kappa, epoch)
            self.writer.add_scalar('Kappa/Val', val_kappa, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # 保存最佳模型
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"✓ 保存最佳模型 (F1: {val_f1:.4f})")

            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.1f}s")
            print(
                f"Train - Loss: {train_loss:.4f}, OA: {train_oa:.4f}, PA: {train_pa:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, IoU: {train_iou:.4f}, Kappa: {train_kappa:.4f}")
            print(
                f"Val   - Loss: {val_loss:.4f}, OA: {val_oa:.4f}, PA: {val_pa:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, IoU: {val_iou:.4f}, Kappa: {val_kappa:.4f}")
            print(f"Best F1: {self.best_val_f1:.4f}")
            print("-" * 60)

        total_time = time.time() - start_time
        print(f"\n训练完成! 总用时: {total_time / 60:.1f}分钟")
        print(f"最佳验证F1: {self.best_val_f1:.4f}")

        # 保存最终模型
        final_model_path = Path('./outputs_train') / 'final_model__adam_one_cross_data2_bce_c3.pth'
        torch.save(self.model.state_dict(), final_model_path)
        print(f"✓ 保存最终模型: {final_model_path}")

        # 保存训练历史
        history_path = './outputs_train/train_history_adam_one_cross_data2_bce_c3.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2, ensure_ascii=False)
        print(f"训练历史已保存到: {history_path}")

        self.writer.close()

    def plot_training_history(self):
        """绘制训练历史曲线"""
        fig = plt.figure(figsize=(18, 12))

        # 损失曲线
        plt.subplot(2, 4, 1)
        plt.plot(self.train_history['train_loss'], label='train_loss', linewidth=2)
        plt.plot(self.train_history['val_loss'], label='val_loss', linewidth=2)
        plt.title('损失曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # F1曲线
        plt.subplot(2, 4, 2)
        plt.plot(self.train_history['train_f1'], label='train_F1', linewidth=2)
        plt.plot(self.train_history['val_f1'], label='val_F1', linewidth=2)
        plt.title('F1曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # IoU曲线
        plt.subplot(2, 4, 3)
        plt.plot(self.train_history['train_iou'], label='train_IoU', linewidth=2)
        plt.plot(self.train_history['val_iou'], label='val_IoU', linewidth=2)
        plt.title('IoU曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # OA曲线
        plt.subplot(2, 4, 4)
        plt.plot(self.train_history['train_oa'], label='train_OA', linewidth=2)
        plt.plot(self.train_history['val_oa'], label='val_OA', linewidth=2)
        plt.title('整体准确率曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('OA')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Kappa曲线
        plt.subplot(2, 4, 5)
        plt.plot(self.train_history['train_kappa'], label='train_Kappa', linewidth=2)
        plt.plot(self.train_history['val_kappa'], label='val_Kappa', linewidth=2)
        plt.title('Kappa系数曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Kappa')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 学习率曲线
        plt.subplot(2, 4, 6)
        plt.plot([self.optimizer.param_groups[0]['lr']] * len(self.train_history['train_loss']),
                 label='学习率', linewidth=2)
        plt.title('学习率曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = "./outputs_train/train_history_cross_data2_bce_c3.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {out_path}")
        plt.close()
