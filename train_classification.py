import random
import argparse
import os
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import logging
from pathlib import Path
import datetime

from PigDataset import PigDataset
from pointnet_cls import get_model
from pointnet_utils import feature_transform_reguliarzer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ================= GPU Augmentation Functions =================

def rotate_point_cloud_z_gpu(batch_data):
    """
    Z-axis rotation on GPU.
    batch_data: [B, C, N]
    """
    B, C, N = batch_data.shape
    device = batch_data.device

    rotation_angle = torch.rand(B, device=device) * 2 * np.pi
    cosval = torch.cos(rotation_angle)
    sinval = torch.sin(rotation_angle)
    zeros = torch.zeros(B, device=device)
    ones = torch.ones(B, device=device)

    R = torch.stack([
        cosval, sinval, zeros,
        -sinval, cosval, zeros,
        zeros, zeros, ones
    ], dim=1).reshape(B, 3, 3)

    xyz = batch_data[:, 0:3, :]
    rotated_xyz = torch.bmm(R, xyz)
    batch_data[:, 0:3, :] = rotated_xyz

    if C >= 7:
        normals = batch_data[:, 4:7, :]
        rotated_normals = torch.bmm(R, normals)
        batch_data[:, 4:7, :] = rotated_normals

    return batch_data


def jitter_point_cloud_gpu(batch_data, sigma=0.001, clip=0.005):
    """
    Jitter on GPU: sigma=0.001, clip=0.005
    """
    B, C, N = batch_data.shape
    device = batch_data.device

    noise = torch.randn(B, 3, N, device=device) * sigma
    noise = torch.clamp(noise, -clip, clip)

    batch_data[:, 0:3, :] += noise
    return batch_data


def shift_point_cloud_gpu(batch_data, shift_range=0.1):
    """
    Shift on GPU
    """
    B, C, N = batch_data.shape
    device = batch_data.device

    shifts = (torch.rand(B, 3, device=device) * 2 * shift_range) - shift_range
    shifts = shifts.unsqueeze(2)

    batch_data[:, 0:3, :] += shifts
    return batch_data


# ===========================================================================

def enforce_adjacent_probs(probs):
    """
    Enforce adjacent probabilities.
    """
    batch_size = probs.shape[0]
    new_probs = probs.clone()

    for i in range(batch_size):
        p_light = probs[i, 0].item()
        p_mid = probs[i, 1].item()
        p_heavy = probs[i, 2].item()

        score_low = p_light + p_mid
        score_high = p_mid + p_heavy

        if score_low >= score_high:
            new_p_light = p_light / (score_low + 1e-8)
            new_p_mid = p_mid / (score_low + 1e-8)
            new_probs[i, :] = torch.tensor([new_p_light, new_p_mid, 0.0])
        else:
            new_p_mid = p_mid / (score_high + 1e-8)
            new_p_heavy = p_heavy / (score_high + 1e-8)
            new_probs[i, :] = torch.tensor([0.0, new_p_mid, new_p_heavy])

    return new_probs


def is_strict_top2_correct(pred_probs, true_label_idx):
    """
    Check if Top-2 is strictly correct.
    """
    top2_indices = torch.topk(pred_probs, k=2).indices.tolist()
    idx1, idx2 = top2_indices[0], top2_indices[1]

    if true_label_idx not in [idx1, idx2]:
        return False
    if abs(idx1 - idx2) > 1:
        return False
    return True


def main():
    setup_seed(16)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_points', type=int, default=16384)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
    parser.add_argument('--outf', type=str,
                        default='./log/classification/fold_x',
                        help='output folder')
    parser.add_argument('--dataset', type=str,
                        default='./dataset_path',
                        help="dataset path")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(args.outf, 'train.log'))
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def log_string(str):
        logger.info(str)
        print(str)

    log_string(str(args))

    csv_path = './weight_soft_top2.csv'

    dataset = PigDataset(root_dir=args.dataset, csv_file=csv_path, split='train', npoints=args.num_points)
    test_dataset = PigDataset(root_dir=args.dataset, csv_file=csv_path, split='test', npoints=args.num_points)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                             drop_last=True)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    classifier = get_model(k=3, channel=7)

    classifier.cuda()
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    print(">>> Training with GPU Augmentation (Fast Mode).")

    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)

    best_strict_top2_acc = 0.0
    best_top1_acc = 0.0

    for epoch in range(args.epoch):
        classifier.train()
        loss_sum = 0

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch} Train'):
            points, soft_target, target, bbox_attr = data

            points = points.transpose(2, 1)
            points, soft_target, target, bbox_attr = points.cuda(), soft_target.cuda(), target.cuda(), bbox_attr.cuda()

            with torch.no_grad():
                points = rotate_point_cloud_z_gpu(points)
                points = jitter_point_cloud_gpu(points)
                points = shift_point_cloud_gpu(points)

            optimizer.zero_grad()
            pred, trans_feat = classifier(points, bbox_attr)

            loss_hard = F.nll_loss(pred, target.long())
            loss_soft = criterion(pred, soft_target)
            loss_cls = 0.3 * loss_hard + 0.7 * loss_soft

            if trans_feat is not None:
                mat_diff_loss = feature_transform_reguliarzer(trans_feat)
                loss = loss_cls + mat_diff_loss * 0.001
            else:
                loss = loss_cls

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        log_string(f'Epoch {epoch} Loss:{loss_sum / len(dataloader):.4f}')
        scheduler.step()

        classifier.eval()
        top1_correct = 0
        strict_top2_correct = 0
        total_samples = 0

        pred_distribution = {0: 0, 1: 0, 2: 0}
        gt_distribution = {0: 0, 1: 0, 2: 0}

        with torch.no_grad():
            for data in testdataloader:
                points, soft_target, hard_target, bbox_attr = data

                points = points.transpose(2, 1)

                points, soft_target, hard_target, bbox_attr = points.cuda(), soft_target.cuda(), hard_target.cuda(), bbox_attr.cuda()

                pred, _ = classifier(points, bbox_attr)
                pred_probs = torch.exp(pred)
                pred_probs_adjusted = enforce_adjacent_probs(pred_probs)

                pred_choice = pred_probs_adjusted.max(1)[1]

                batch_size = points.size(0)
                total_samples += batch_size
                top1_correct += pred_choice.eq(hard_target).cpu().sum().item()

                pred_np = pred_choice.cpu().numpy()
                gt_np = hard_target.cpu().numpy()

                for p in pred_np:
                    pred_distribution[int(p)] += 1
                for g in gt_np:
                    gt_distribution[int(g)] += 1

                for b in range(batch_size):
                    single_pred_probs = pred_probs_adjusted[b]
                    single_true_idx = hard_target[b].item()

                    if is_strict_top2_correct(single_pred_probs, single_true_idx):
                        strict_top2_correct += 1

        top1_acc = top1_correct / total_samples
        strict_top2_acc = strict_top2_correct / total_samples

        log_string(f'Epoch {epoch} Test Results:')
        log_string(f'  Top-1 Accuracy:{top1_acc:.4f}')
        log_string(f'  Strict Top2 Accuracy:{strict_top2_acc:.4f}')
        log_string(f'  > Pred Distribution:{pred_distribution} (Total:{total_samples})')
        log_string(f'  > GT Distribution:  {gt_distribution}')

        is_new_best_top1 = top1_acc > best_top1_acc
        is_new_best_top2 = strict_top2_acc > best_strict_top2_acc

        if is_new_best_top1:
            best_top1_acc = top1_acc
        if is_new_best_top2:
            best_strict_top2_acc = strict_top2_acc

        if is_new_best_top1 or is_new_best_top2:
            log_string(f'  >>> Found New Best Model (Top1: {is_new_best_top1}, Top2: {is_new_best_top2})')

            savepath_t1 = os.path.join(args.outf, f'epoch{epoch}_best_top1_{top1_acc:.4f}.pth')
            torch.save(classifier.state_dict(), savepath_t1)

            savepath_t2 = os.path.join(args.outf, f'epoch{epoch}_best_top2_{strict_top2_acc:.4f}.pth')
            torch.save(classifier.state_dict(), savepath_t2)

            log_string(f'      Saved: {os.path.basename(savepath_t1)} & {os.path.basename(savepath_t2)}')


if __name__ == '__main__':
    main()