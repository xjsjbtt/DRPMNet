#!/usr/bin/env python3
"""
DINOv3图像分割完整训练和评估流程
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
from data_loader_new import LandslideDataset
import argparse

from datanew import create_data_loaders
from seg_dinov3_res18 import create_model

from uent import UNet_Multimodal

# from train_segmentation_one_loss import SegmentationTrainer
from train_segmentation_two_loss import SegmentationTrainer
from evaluate_segmentation_first import SegmentationEvaluator, SegmentationVisualizer


def train(device, data_dir, batch_size, image_size, pretrained_path, lr, epochs, num_classes=0, experiment_name=None):
    print("\n" + "=" * 60)
    print("开始训练阶段")
    print("=" * 60)  # 创建数据加载器 print("创建数据加载器...")

    train_loader, val_loader, _, _, _ = create_data_loaders(data_dir=str(data_dir),
                                                            batch_size=batch_size,
                                                            num_workers=2,
                                                            image_size=image_size)
    # 创建模型 print("创建DINOv3分割模型...")
    model = create_model(num_classes, pretrained_path=str(pretrained_path) if pretrained_path else None)
    # model=UNet_Multimodal(n_channels_img=3,n_channels_dem=1,num_classes=1)
    # 创建训练器,参数均在上面设置好
    trainer = SegmentationTrainer(model=model,
                                  train_loader=train_loader,
                                  val_loader=val_loader,
                                  device=device,
                                  lr=lr, epoch=epochs, experiment_name=experiment_name)
    # 开始训练
    start_time = time.time()
    trainer.train(num_epochs=epochs)
    training_time = time.time() - start_time
    print(f"\n训练完成! 用时: {training_time / 60:.1f}分钟")

    print(f"最佳验证IoU: {trainer.best_val_f1:.4f}")
    # 绘制训练历史
    trainer.plot_training_history()
    print("训练任务完成")
    print("=" * 60)
    print("生成的文件:")
    print("- best_model.pth: 最佳模型权重")
    print("- training_history.json: 训练历史")
    print("- training_history.png")


def eval(mode, device, data_dir, batch_size, image_size, pretrained_path, epochs, num_classes, experiment_name=None):
    print("\n" + "=" * 60)
    print("开始评估阶段")
    print("=" * 60)

    # 检查模型文件
    model_path = f'outputs_train/{experiment_name}/best_model.pth'
    if not Path(model_path).exists():
        print(f"模型文件不存在: {model_path}")
        if mode == 'train':
            print("请先运行训练模式")
            return
        else:
            print("跳过评估阶段")
            return

    # 创建模型并加载权重
    print("加载训练好的模型...")
    model = create_model(num_classes, pretrained_path=str(pretrained_path) if pretrained_path else None)
    # model=UNet_Multimodal(n_channels_img=3,n_channels_dem=1,num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 创建数据加载器
    _, val_loader, test_loader, val_dataset, test_dataset = create_data_loaders(
        data_dir=str(data_dir),
        batch_size=batch_size,
        num_workers=2,
        image_size=image_size
    )

    # 定义统一保存目录
    save_dir = Path('./outputs') / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 评估模型
    print("评估模型性能...")
    evaluator = SegmentationEvaluator(model, test_loader, device)
    results = evaluator.evaluate()
    evaluator.print_results()
    evaluator.save_results(save_path=save_dir / 'evaluation_results.json')
    # 生成详细报告
    evaluator.generate_report(save_path=save_dir / 'evaluation_report.txt')

    # 可视化结果
    print("生成可视化结果...")
    visualizer = SegmentationVisualizer(model, test_dataset, device)
    # 预测可视化
    """
    visualizer.visualize_predictions(
        num_samples=33,
        save_path=save_dir / 'prediction_results.png'
    )
    # 混淆矩阵
    cm = np.array(results['confusion_matrix'])
    visualizer.create_confusion_matrix_plot(cm, save_path=save_dir / 'confusion_matrix.png')
    # 概率热力图
    sample = test_dataset[0]
    visualizer.create_probability_heatmap(
        sample['image'], sample['dem'], save_path=save_dir / 'probability_heatmap.png'
    )
    """
    visualizer.export_per_sample_results("./outputs/per_image_results")

    print("评估任务完成")
    print("")
    print("- prediction_results.png: 预测结果可视化")
    print("- confusion_matrix.png: 混淆矩阵")
    print("- probability_heatmap.png: 概率热力图")
    print("- evaluation_report.txt: 详细评估报告")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DINOv3图像分割训练和评估')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'both'],
                        help='运行模式: train(训练), eval(评估), both(训练+评估)')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--image_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--data_dir', type=str, default='./newdata_split', help='数据目录')
    parser.add_argument('--pretrained_path', type=str,
                        default='weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
                        help='预训练模型路径')
    parser.add_argument(
        '--experiment_name', type=str, default='model_18_newdata',
        help='实验名称/子文件夹名称，训练结果将保存到 outputs_train/experiment_name 或评估结果 outputs/experiment_name 下')

    parser.add_argument('--num_classes', type=int, default=1, help='模型分类数量')

    args = parser.parse_args()
    print("===========item start===========")
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")

    # 检查数据目录
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return

    # 检查预训练模型
    pretrained_path = Path(args.pretrained_path)

    if not pretrained_path.exists():
        print(f"预训练模型不存在: {pretrained_path}")
        print("将使用ImageNet预训练权重")
        pretrained_path = None
    if args.mode in ['train']:
        train(device, data_dir, args.batch_size, args.image_size, args.pretrained_path, args.lr, args.epochs,
              args.num_classes, args.experiment_name)
    elif args.mode in ['eval']:
        eval(args.mode, device, data_dir, args.batch_size, args.image_size, pretrained_path, args.epochs,
             args.num_classes, args.experiment_name)
    else:
        train(device, data_dir, args.batch_size, args.image_size, args.pretrained_path, args.lr, args.epochs,
              args.num_classes, args.experiment_name)
        eval(args.mode, device, data_dir, args.batch_size, args.image_size, pretrained_path, args.epochs,
             args.num_classes, args.experiment_name)
    print("=========item over==========")


if __name__ == "__main__":
    main()
