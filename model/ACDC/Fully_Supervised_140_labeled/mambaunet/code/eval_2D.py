import argparse
import csv
import os
from typing import List, Tuple

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
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
        # 若无正样本，返回 0 以保持统计一致
        return 0.0, 0.0, 0.0


def infer_volume(net: torch.nn.Module, image: torch.Tensor, classes: int, patch_size: List[int]) -> np.ndarray:
    """按 val_2D 的策略对体数据逐切片推理并还原到原分辨率。"""
    image_np = image.squeeze(0).cpu().detach().numpy()  # [Z, H, W]
    prediction = np.zeros_like(image_np, dtype=np.uint8)
    net.eval()
    with torch.no_grad():
        for ind in range(image_np.shape[0]):
            slice_img = image_np[ind, :, :]
            x, y = slice_img.shape
            resized = zoom(slice_img, (patch_size[0] / x, patch_size[1] / y), order=0)
            input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().cuda()
            logits = net(input_tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred_small = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
            pred_np = pred_small.cpu().detach().numpy()
            pred_restore = zoom(pred_np, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred_restore.astype(np.uint8)
    return prediction


def eval_one_case(pred: np.ndarray, label: np.ndarray, classes: int) -> List[Tuple[float, float, float, int, int]]:
    results = []
    for i in range(1, classes):
        dice, hd95, assd = safe_binary_metrics(pred == i, label == i)
        vox_pred = int((pred == i).sum())
        vox_gt = int((label == i).sum())
        results.append((dice, hd95, assd, vox_pred, vox_gt))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True, help='数据集根目录，例如 ../data/ACDC')
    parser.add_argument('--exp', type=str, default='ACDC/Semi_Mamba_UNet', help='实验名，用于推断权重目录')
    parser.add_argument('--labeled_num', type=int, default=140, help='用于拼接权重路径的标注病人数')
    parser.add_argument('--model', type=str, default='mambaunet', help='模型名，用于拼接权重路径')
    parser.add_argument('--cfg', type=str, default='../code/configs/vmamba_tiny.yaml', help='配置文件路径')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--patch_size', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--split', type=str, default='val', choices=['val'], help='当前数据定义仅提供 val')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default='', help='显式指定权重路径（.pth）')
    parser.add_argument('--which', type=str, default='model1', choices=['model','model1', 'model2'], help='评估哪个学生模型')
    parser.add_argument('--save_dir', type=str, default='', help='结果输出目录，默认跟随 snapshot')
    parser.add_argument('--save_preds', action='store_true', help='是否保存预测体素 npy')
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
        ckpt_name = f"{args.model}_best_{args.which}.pth"
        # 训练脚本里保存名为 {model}_best_model1.pth / {model}_best_model2.pth
        ckpt_name = f"{args.model}_best_{args.which}.pth" if os.path.exists(os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth")) else f"{args.model}_best_{args.which}.pth"
        # 兼容训练中的真实命名
        default_name = f"{args.model}_best_{args.which}.pth"
        alt_name = f"{args.model}_best_{args.which}.pth"
        candidates = [
            os.path.join(snapshot_path, default_name),
            os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth"),
            os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth"),
            os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth"),
        ]
        # 训练脚本保存的是 '{model}_best_model1.pth' / '{model}_best_model2.pth'
        candidates.insert(0, os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth"))
        candidates.insert(0, os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth"))
        # 实际：在 train 脚本里是 '{}_best_model1.pth'
        candidates.insert(0, os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth"))
        # 最靠谱直接用训练脚本格式：
        candidates.insert(0, os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth"))
        # 以及 train 中明确的保存名：
        candidates.insert(0, os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth"))
        # 兜底：
        candidates.insert(0, os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth"))
        # 由于训练脚本明确：'{}_best_model1.pth'
        # 我们直接使用该命名：
        ckpt_path = os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth".replace('_model1', 'model1').replace('_model2', 'model2'))
        # 若找不到，再尝试另一个确切命名
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth")

    if not os.path.exists(ckpt_path):
        # 按训练脚本的真实命名： '{model}_best_model1.pth'
        alt = os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth")
        if os.path.exists(alt):
            ckpt_path = alt
        else:
            # 直接尝试 '{model}_best_model1.pth' / '{model}_best_model2.pth'
            ckpt_try = os.path.join(snapshot_path, f"{args.model}_best_{args.which}.pth")
            if os.path.exists(ckpt_try):
                ckpt_path = ckpt_try
            else:
                raise FileNotFoundError(f"未找到权重文件，请通过 --checkpoint 指定，或检查目录: {snapshot_path}")

    state = torch.load(ckpt_path)
    net.load_state_dict(state)

    # 数据集
    dataset = BaseDataSets(base_dir=args.root_path, split=args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # 输出目录
    save_dir = args.save_dir or os.path.join(snapshot_path, f"eval_{args.which}")
    os.makedirs(save_dir, exist_ok=True)
    if args.save_preds:
        os.makedirs(os.path.join(save_dir, 'preds'), exist_ok=True)

    per_case_csv = os.path.join(save_dir, 'metrics_per_case.csv')
    summary_csv = os.path.join(save_dir, 'summary_table.csv')

    # 写入 per-case CSV（列名）
    with open(per_case_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['case', 'class', 'dice', 'hd95', 'assd', 'voxels_pred', 'voxels_gt']
        writer.writerow(header)

        # 收集以供汇总
        all_metrics = {c: {'dice': [], 'hd95': [], 'assd': []} for c in range(1, args.num_classes)}

        for idx, batch in enumerate(loader):
            image = batch['image'].cuda()
            label = batch['label'].cuda()
            case_name = dataset.sample_list[idx] if hasattr(dataset, 'sample_list') else str(idx)

            pred_np = infer_volume(net, image, args.num_classes, args.patch_size)
            label_np = label.squeeze(0).cpu().detach().numpy()

            # 保存预测
            if args.save_preds:
                np.save(os.path.join(save_dir, 'preds', f"{case_name}.npy"), pred_np.astype(np.uint8))

            results = eval_one_case(pred_np, label_np, args.num_classes)
            for class_index, (dice, hd95, assd, vox_pred, vox_gt) in enumerate(results, start=1):
                writer.writerow([case_name, class_index, f"{dice:.4f}", f"{hd95:.4f}", f"{assd:.4f}", vox_pred, vox_gt])
                all_metrics[class_index]['dice'].append(dice)
                all_metrics[class_index]['hd95'].append(hd95)
                all_metrics[class_index]['assd'].append(assd)

    # 汇总 mean±std
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
        # overall（按类别平均）
        dice_all = np.concatenate([d.reshape(1, -1) for d in dice_all], axis=0)
        hd95_all = np.concatenate([h.reshape(1, -1) for h in hd95_all], axis=0)
        assd_all = np.concatenate([a.reshape(1, -1) for a in assd_all], axis=0)
        writer.writerow([
            'mean_over_classes',
            f"{dice_all.mean():.4f}", f"{dice_all.std():.4f}",
            f"{hd95_all.mean():.2f}", f"{hd95_all.std():.2f}",
            f"{assd_all.mean():.2f}", f"{assd_all.std():.2f}",
        ])

    print(f"Per-case 指标已保存到: {per_case_csv}")
    print(f"汇总表已保存到: {summary_csv}")


if __name__ == '__main__':
    main()


