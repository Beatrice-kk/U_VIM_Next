import argparse
import csv
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom, binary_erosion, binary_dilation, find_objects
from torch.utils.data import DataLoader

from config import get_config
from dataloaders.dataset import BaseDataSets
from networks.vision_mamba import MambaUnet as ViM_seg


def safe_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = float(metric.binary.dc(pred, gt))
        try:
            hd95 = float(metric.binary.hd95(pred, gt))
        except Exception:
            hd95 = 0.0
        try:
            assd = float(metric.binary.asd(pred, gt))
        except Exception:
            assd = 0.0
        return dice, hd95, assd
    else:
        return 0.0, 0.0, 0.0


def compute_boundary_metrics(pred: np.ndarray, gt: np.ndarray, erosion_size: int = 3) -> Dict[str, float]:
    """计算边界区域和内部区域的指标，用于诊断浅层vs深层特征问题
    
    边界区域：需要精确的浅层特征（边缘检测）
    内部区域：需要深层语义理解（区域一致性）
    """
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    
    if gt_bin.sum() == 0:
        return {
            'boundary_dice': 0.0,
            'boundary_hd95': 0.0,
            'inner_dice': 0.0,
            'boundary_voxels': 0,
            'inner_voxels': 0
        }
    
    # 获取边界区域（通过腐蚀的差异）
    # 使用3x3的结构元素进行腐蚀
   # structure = np.ones((erosion_size, erosion_size), dtype=np.uint8)
    structure = np.ones((gt_bin.ndim) * (erosion_size,), dtype=np.uint8)  #自适应维度
    gt_eroded = binary_erosion(gt_bin, structure=structure).astype(np.uint8)
    gt_boundary = gt_bin - gt_eroded
    
    # 内部区域
    gt_inner = gt_eroded
    
    # 边界区域指标
    boundary_dice = 0.0
    boundary_hd95 = 0.0
    if gt_boundary.sum() > 0:
        pred_boundary = pred_bin * gt_boundary
        if pred_boundary.sum() > 0 or gt_boundary.sum() > 0:
            boundary_dice, boundary_hd95, _ = safe_binary_metrics(pred_boundary, gt_boundary)
    
    # 内部区域指标
    inner_dice = 0.0
    if gt_inner.sum() > 0:
        pred_inner = pred_bin * gt_inner
        if pred_inner.sum() > 0 and gt_inner.sum() > 0:
            inner_dice, _, _ = safe_binary_metrics(pred_inner, gt_inner)
    
    return {
        'boundary_dice': boundary_dice,
        'boundary_hd95': boundary_hd95,
        'inner_dice': inner_dice,
        'boundary_voxels': int(gt_boundary.sum()),
        'inner_voxels': int(gt_inner.sum())
    }


def compute_size_based_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """根据目标大小分类评估，用于诊断不同尺度的特征提取问题
    
    小目标：主要依赖浅层特征（边缘检测）
    大目标：主要依赖深层语义理解（区域一致性）
    """
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)
    
    # 连通域分析（3D体积）
    from scipy.ndimage import label
    labeled_gt, num_objects = label(gt_bin)
    
    small_dice = []
    medium_dice = []
    large_dice = []
    
    for obj_id in range(1, num_objects + 1):
        obj_mask = (labeled_gt == obj_id)
        volume = obj_mask.sum()  # 3D体积中的体素数
        
        obj_gt = gt_bin * obj_mask
        obj_pred = pred_bin * obj_mask
        
        if obj_gt.sum() > 0:
            dice, _, _ = safe_binary_metrics(obj_pred, obj_gt)
            
            # 根据体素数分类：小(<10000), 中(10000-100000), 大(>100000)
            # 这些阈值针对3D医学图像体积
            if volume < 10000:
                small_dice.append(dice)
            elif volume < 100000:
                medium_dice.append(dice)
            else:
                large_dice.append(dice)
    
    return {
        'small_dice_mean': np.mean(small_dice) if small_dice else 0.0,
        'small_count': len(small_dice),
        'medium_dice_mean': np.mean(medium_dice) if medium_dice else 0.0,
        'medium_count': len(medium_dice),
        'large_dice_mean': np.mean(large_dice) if large_dice else 0.0,
        'large_count': len(large_dice),
    }


def infer_volume_with_intermediate_features(net: torch.nn.Module, image: torch.Tensor, 
                                           classes: int, patch_size: List[int],
                                           return_intermediate: bool = False) -> Tuple[np.ndarray, Dict]:
    """推理并返回中间特征，用于分析各层贡献"""
    image_np = image.squeeze(0).cpu().detach().numpy()  # [Z, H, W]
    prediction = np.zeros_like(image_np, dtype=np.uint8)
    
    # 存储中间特征（用于分析）
    intermediate_features = {}
    if return_intermediate:
        intermediate_features = {
            'encoder_layers': [],
            'decoder_layers': []
        }
    
    net.eval()
    with torch.no_grad():
        for ind in range(image_np.shape[0]):
            slice_img = image_np[ind, :, :]
            x, y = slice_img.shape
            resized = zoom(slice_img, (patch_size[0] / x, patch_size[1] / y), order=0)
            input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().cuda()
            
            # 获取编码器中间特征
            if return_intermediate:
                x_feat, x_downsample = net.mamba_unet.forward_features(input_tensor)
                intermediate_features['encoder_layers'].append({
                    'layer_0': x_downsample[0].cpu().numpy() if len(x_downsample) > 0 else None,
                    'layer_1': x_downsample[1].cpu().numpy() if len(x_downsample) > 1 else None,
                    'layer_2': x_downsample[2].cpu().numpy() if len(x_downsample) > 2 else None,
                    'layer_3': x_downsample[3].cpu().numpy() if len(x_downsample) > 3 else None,
                    'bottleneck': x_feat.cpu().numpy()
                })
            
            logits = net(input_tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred_small = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
            pred_np = pred_small.cpu().detach().numpy()
            pred_restore = zoom(pred_np, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred_restore.astype(np.uint8)
    
    return prediction, intermediate_features


def infer_volume(net: torch.nn.Module, image: torch.Tensor, classes: int, patch_size: List[int]) -> np.ndarray:
    """按 val_2D 的策略对体数据逐切片推理并还原到原分辨率。"""
    pred, _ = infer_volume_with_intermediate_features(net, image, classes, patch_size, return_intermediate=False)
    return pred


def eval_one_case_comprehensive(pred: np.ndarray, label: np.ndarray, classes: int) -> Dict:
    """综合评估单个病例，包括边界、内部、大小分类等指标"""
    results = {
        'class_metrics': [],
        'boundary_metrics': [],
        'size_metrics': []
    }
    
    for i in range(1, classes):
        # 基础指标
        dice, hd95, assd = safe_binary_metrics(pred == i, label == i)
        vox_pred = int((pred == i).sum())
        vox_gt = int((label == i).sum())
        
        # 边界vs内部指标（诊断浅层特征问题）
        boundary_metrics = compute_boundary_metrics(pred == i, label == i)
        
        # 大小分类指标（诊断不同尺度特征问题）
        size_metrics = compute_size_based_metrics(pred == i, label == i)
        
        results['class_metrics'].append({
            'class': i,
            'dice': dice,
            'hd95': hd95,
            'assd': assd,
            'vox_pred': vox_pred,
            'vox_gt': vox_gt
        })
        
        results['boundary_metrics'].append({
            'class': i,
            **boundary_metrics
        })
        
        results['size_metrics'].append({
            'class': i,
            **size_metrics
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='诊断评估：分析模型baseline的浅层vs深层特征问题')
    parser.add_argument('--root_path', type=str, required=True, help='数据集根目录')
    parser.add_argument('--exp', type=str, default='ACDC/Semi_Mamba_UNet', help='实验名')
    parser.add_argument('--labeled_num', type=int, default=140, help='标注病人数')
    parser.add_argument('--model', type=str, default='mambaunet', help='模型名')
    parser.add_argument('--cfg', type=str, default='../code/configs/vmamba_tiny.yaml', help='配置文件路径')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--patch_size', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--split', type=str, default='val', choices=['val'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default='', help='显式指定权重路径')
    parser.add_argument('--which', type=str, default='model1', choices=['model','model1', 'model2'])
    parser.add_argument('--save_dir', type=str, default='', help='结果输出目录')
    parser.add_argument('--save_preds', action='store_true', help='是否保存预测体素')
    args = parser.parse_args()

    # 配置与模型
    cfg = get_config(args)
    net = ViM_seg(cfg, img_size=args.patch_size, num_classes=args.num_classes).cuda()
    net.load_from(cfg)

    # 推断权重路径
    snapshot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', f"{args.exp}", args.model))
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"未找到权重文件: {ckpt_path}")

    state = torch.load(ckpt_path)
    #net.load_state_dict(state)
    net.load_state_dict(state, strict=False)

    # 数据集
    dataset = BaseDataSets(base_dir=args.root_path, split=args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # 输出目录
    save_dir = args.save_dir or os.path.join(snapshot_path, f"eval_diagnostic_{args.which}")
    os.makedirs(save_dir, exist_ok=True)
    if args.save_preds:
        os.makedirs(os.path.join(save_dir, 'preds'), exist_ok=True)

    # CSV文件路径
    per_case_csv = os.path.join(save_dir, 'metrics_per_case.csv')
    boundary_csv = os.path.join(save_dir, 'boundary_analysis.csv')
    size_csv = os.path.join(save_dir, 'size_analysis.csv')
    summary_csv = os.path.join(save_dir, 'summary_table.csv')
    diagnostic_csv = os.path.join(save_dir, 'diagnostic_summary.csv')

    # 初始化汇总数据
    all_metrics = {c: {'dice': [], 'hd95': [], 'assd': []} for c in range(1, args.num_classes)}
    all_boundary = {c: {'boundary_dice': [], 'boundary_hd95': [], 'inner_dice': []} for c in range(1, args.num_classes)}
    all_size = {c: {'small_dice': [], 'medium_dice': [], 'large_dice': []} for c in range(1, args.num_classes)}

    # 写入 per-case CSV
    with open(per_case_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['case', 'class', 'dice', 'hd95', 'assd', 'voxels_pred', 'voxels_gt']
        writer.writerow(header)

        with open(boundary_csv, 'w', newline='') as f_boundary:
            writer_boundary = csv.writer(f_boundary)
            writer_boundary.writerow(['case', 'class', 'boundary_dice', 'boundary_hd95', 'inner_dice', 
                                     'boundary_voxels', 'inner_voxels'])

            with open(size_csv, 'w', newline='') as f_size:
                writer_size = csv.writer(f_size)
                writer_size.writerow(['case', 'class', 'small_dice', 'small_count', 
                                     'medium_dice', 'medium_count', 'large_dice', 'large_count'])

                for idx, batch in enumerate(loader):
                    image = batch['image'].cuda()
                    label = batch['label'].cuda()
                    case_name = dataset.sample_list[idx] if hasattr(dataset, 'sample_list') else str(idx)

                    pred_np = infer_volume(net, image, args.num_classes, args.patch_size)
                    label_np = label.squeeze(0).cpu().detach().numpy()

                    # 保存预测
                    if args.save_preds:
                        np.save(os.path.join(save_dir, 'preds', f"{case_name}.npy"), pred_np.astype(np.uint8))

                    # 综合评估
                    results = eval_one_case_comprehensive(pred_np, label_np, args.num_classes)

                    # 写入各类指标
                    for class_idx, class_metric in enumerate(results['class_metrics'], start=1):
                        writer.writerow([
                            case_name, class_metric['class'],
                            f"{class_metric['dice']:.4f}", f"{class_metric['hd95']:.4f}", f"{class_metric['assd']:.4f}",
                            class_metric['vox_pred'], class_metric['vox_gt']
                        ])
                        all_metrics[class_metric['class']]['dice'].append(class_metric['dice'])
                        all_metrics[class_metric['class']]['hd95'].append(class_metric['hd95'])
                        all_metrics[class_metric['class']]['assd'].append(class_metric['assd'])

                    # 写入边界分析
                    for boundary_metric in results['boundary_metrics']:
                        writer_boundary.writerow([
                            case_name, boundary_metric['class'],
                            f"{boundary_metric['boundary_dice']:.4f}",
                            f"{boundary_metric['boundary_hd95']:.4f}",
                            f"{boundary_metric['inner_dice']:.4f}",
                            boundary_metric['boundary_voxels'],
                            boundary_metric['inner_voxels']
                        ])
                        all_boundary[boundary_metric['class']]['boundary_dice'].append(boundary_metric['boundary_dice'])
                        all_boundary[boundary_metric['class']]['boundary_hd95'].append(boundary_metric['boundary_hd95'])
                        all_boundary[boundary_metric['class']]['inner_dice'].append(boundary_metric['inner_dice'])

                    # 写入大小分析
                    for size_metric in results['size_metrics']:
                        writer_size.writerow([
                            case_name, size_metric['class'],
                            f"{size_metric['small_dice_mean']:.4f}", size_metric['small_count'],
                            f"{size_metric['medium_dice_mean']:.4f}", size_metric['medium_count'],
                            f"{size_metric['large_dice_mean']:.4f}", size_metric['large_count']
                        ])
                        if size_metric['small_count'] > 0:
                            all_size[size_metric['class']]['small_dice'].append(size_metric['small_dice_mean'])
                        if size_metric['medium_count'] > 0:
                            all_size[size_metric['class']]['medium_dice'].append(size_metric['medium_dice_mean'])
                        if size_metric['large_count'] > 0:
                            all_size[size_metric['class']]['large_dice'].append(size_metric['large_dice_mean'])

    # 汇总表
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'dice_mean', 'dice_std', 'hd95_mean', 'hd95_std', 'assd_mean', 'assd_std'])
        dice_all = []
        hd95_all = []
        assd_all = []
        for c in range(1, args.num_classes):
            dice_arr = np.array(all_metrics[c]['dice'], dtype=np.float32)
            hd95_arr = np.array(all_metrics[c]['hd95'], dtype=np.float32)
            assd_arr = np.array(all_metrics[c]['assd'], dtype=np.float32)
            writer.writerow([
                c,
                f"{dice_arr.mean():.4f}", f"{dice_arr.std():.4f}",
                f"{hd95_arr.mean():.2f}", f"{hd95_arr.std():.2f}",
                f"{assd_arr.mean():.2f}", f"{assd_arr.std():.2f}",
            ])
            dice_all.append(dice_arr)
            hd95_all.append(hd95_arr)
            assd_all.append(assd_arr)
        
        dice_all = np.concatenate([d.reshape(1, -1) for d in dice_all], axis=0)
        hd95_all = np.concatenate([h.reshape(1, -1) for h in hd95_all], axis=0)
        assd_all = np.concatenate([a.reshape(1, -1) for a in assd_all], axis=0)
        writer.writerow([
            'mean_over_classes',
            f"{dice_all.mean():.4f}", f"{dice_all.std():.4f}",
            f"{hd95_all.mean():.2f}", f"{hd95_all.std():.2f}",
            f"{assd_all.mean():.2f}", f"{assd_all.std():.2f}",
        ])

    # 诊断汇总表（关键指标）
    with open(diagnostic_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'boundary_dice_mean', 'inner_dice_mean', 'boundary_inner_gap', 
                         'small_dice_mean', 'medium_dice_mean', 'large_dice_mean', 
                         'diagnosis'])
        
        for c in range(1, args.num_classes):
            boundary_dice_arr = np.array(all_boundary[c]['boundary_dice'], dtype=np.float32)
            inner_dice_arr = np.array(all_boundary[c]['inner_dice'], dtype=np.float32)
            
            boundary_dice_mean = boundary_dice_arr.mean()
            inner_dice_mean = inner_dice_arr.mean()
            gap = inner_dice_mean - boundary_dice_mean
            
            small_dice_mean = np.mean(all_size[c]['small_dice']) if all_size[c]['small_dice'] else 0.0
            medium_dice_mean = np.mean(all_size[c]['medium_dice']) if all_size[c]['medium_dice'] else 0.0
            large_dice_mean = np.mean(all_size[c]['large_dice']) if all_size[c]['large_dice'] else 0.0
            
            # 诊断结论
            diagnosis = []
            if gap > 0.15:  # 边界明显差于内部
                diagnosis.append("浅层特征提取不足（边界精度差）")
            if inner_dice_mean < 0.7:  # 内部区域也差
                diagnosis.append("深层语义理解不足")
            if small_dice_mean < medium_dice_mean - 0.1:  # 小目标表现差
                diagnosis.append("小目标检测能力弱（浅层特征问题）")
            if large_dice_mean < medium_dice_mean - 0.1:  # 大目标表现差
                diagnosis.append("大目标分割能力弱（深层语义问题）")
            
            if not diagnosis:
                diagnosis = ["模型表现均衡"]
            
            writer.writerow([
                c,
                f"{boundary_dice_mean:.4f}",
                f"{inner_dice_mean:.4f}",
                f"{gap:.4f}",
                f"{small_dice_mean:.4f}",
                f"{medium_dice_mean:.4f}",
                f"{large_dice_mean:.4f}",
                "; ".join(diagnosis)
            ])

    print(f"Per-case 指标已保存到: {per_case_csv}")
    print(f"边界分析已保存到: {boundary_csv}")
    print(f"大小分析已保存到: {size_csv}")
    print(f"汇总表已保存到: {summary_csv}")
    print(f"诊断汇总已保存到: {diagnostic_csv}")
    print("\n=== 诊断说明 ===")
    print("1. boundary_dice < inner_dice: 浅层特征提取不足（边界精度差）")
    print("2. inner_dice 较低: 深层语义理解不足")
    print("3. 小目标dice < 大目标dice: 浅层特征提取问题")
    print("4. 大目标dice < 小目标dice: 深层语义理解问题")


if __name__ == '__main__':
    main()

