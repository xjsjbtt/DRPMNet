import numpy as np


def ConfusionMatrix(num_classes, pres, gts):
    def __get_hist(pre, gt):
        pre = pre.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        mask = (gt >= 0) & (gt < num_classes)
        label = num_classes * gt[mask].astype(int) + pre[mask].astype(int)
        hist = np.bincount(label, minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    cm = np.zeros((num_classes, num_classes))
    for lt, lp in zip(gts, pres):
        cm += __get_hist(lt.flatten(), lp.flatten())
    return cm


def get_score(confusionMatrix):
    TN = confusionMatrix[0, 0]
    FP = confusionMatrix[0, 1]
    FN = confusionMatrix[1, 0]
    TP = confusionMatrix[1, 1]

    eps = np.finfo(np.float32).eps

    # 1. 精确率
    # 类别0（背景）精确率 = TN / (TN + FP)
    # 类别1（前景）精确率 = TP / (TP + FP)
    precision_0 = TN / (TN + FP + eps)
    precision_1 = TP / (TP + FP + eps)
    precision = np.array([precision_0, precision_1])

    # 2. 召回率
    # 类别0召回率 = TN / (TN + FN)
    # 类别1召回率 = TP / (TP + FN)
    recall_0 = TN / (TN + FN + eps)
    recall_1 = TP / (TP + FN + eps)
    recall = np.array([recall_0, recall_1])

    # 3. F1分数
    f1score = 2 * precision * recall / (precision + recall + eps)

    # 4. IoU（交并比）
    # 类别0 IoU = TN / (TN + FP + FN)
    # 类别1 IoU = TP / (TP + FP + FN)
    iou_0 = TN / (TN + FP + FN + eps)
    iou_1 = TP / (TP + FP + FN + eps)
    iou = np.array([iou_0, iou_1])

    # 5. Kappa系数
    total = TN + FP + FN + TP
    po = (TN + TP) / (total + eps)  # 总体准确率

    # 预期一致概率
    # pe = (真实0的数量 * 预测0的数量 + 真实1的数量 * 预测1的数量) / 总样本数^2
    real_0 = TN + FP  # 真实为0的数量
    real_1 = FN + TP  # 真实为1的数量
    pred_0 = TN + FN  # 预测为0的数量
    pred_1 = FP + TP  # 预测为1的数量

    pe = (real_0 * pred_0 + real_1 * pred_1) / (total ** 2 + eps)
    kc = (po - pe) / (1 - pe + eps)

    return precision, recall, f1score, iou, kc


def get_score_sum(confusionMatrix):
    num_classes = confusionMatrix.shape[0]
    precision, recall, f1score, iou, kc = get_score(confusionMatrix)
    cls_precision = dict(zip(['precision_' + str(i) for i in range(num_classes)], precision))
    cls_recall = dict(zip(['recall_' + str(i) for i in range(num_classes)], recall))
    cls_f1 = dict(zip(['f1_' + str(i) for i in range(num_classes)], f1score))
    cls_iou = dict(zip(['iou_' + str(i) for i in range(num_classes)], iou))
    return cls_precision, cls_recall, cls_f1, cls_iou, kc


