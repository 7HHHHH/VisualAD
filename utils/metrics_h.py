from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
from tabulate import tabulate
import numpy as np
import logging
from skimage import measure

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = np.min(amaps), np.max(amaps)
    delta = (max_th - min_th) / max_step
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[:] = amaps > th
        pro = []
        for i in range(len(amaps)):
            mask = masks[i]
            amap = binary_amaps[i]
            if np.sum(mask) == 0:
                continue  # Skip if no anomaly region
            labeled_mask = measure.label(mask)
            regions = measure.regionprops(labeled_mask)
            for region in regions:
                coords = region.coords
                tp = np.sum(amap[coords[:, 0], coords[:, 1]])
                pro.append(tp / region.area)
        avg_pro = np.mean(pro) if pro else 0
        pros.append(avg_pro)
        fp = np.sum(binary_amaps * (1 - masks))
        total_fp_pixels = np.sum(1 - masks)
        fpr = fp / total_fp_pixels if total_fp_pixels != 0 else 0
        fprs.append(fpr)
    pros, fprs = np.array(pros), np.array(fprs)
    valid = fprs <= expect_fpr
    if not np.any(valid):
        return 0.0
    fpr_valid = fprs[valid]
    pro_valid = pros[valid]
    if len(fpr_valid) < 2:
        return 0.0
    fpr_norm = (fpr_valid - fpr_valid.min()) / (fpr_valid.max() - fpr_valid.min())
    return auc(fpr_norm, pro_valid)

def compute_metrics(results, obj_list, logger):
    table_ls = []
    for obj in obj_list:
        obj_data = results[obj]
        # Pixel-level data
        gt_px, pr_px = [], []
        masks, amaps = [], []
        for mask_batch, anomaly_map_batch in zip(obj_data['imgs_masks'], obj_data['anomaly_maps']):
            mask_np = mask_batch.squeeze().cpu().numpy()
            amap_np = anomaly_map_batch.squeeze().cpu().numpy()
            gt_px.extend(mask_np.flatten())
            pr_px.extend(amap_np.flatten())
            masks.append(mask_np)
            amaps.append(amap_np)
        gt_px = np.array(gt_px)
        pr_px = np.array(pr_px)
        masks = np.stack(masks)
        amaps = np.stack(amaps)
        # Sample-level data
        gt_sp = np.array(obj_data['gt_sp'])
        pr_sp = np.array(obj_data['pr_sp'])
        
        # Calculate metrics
        pixel_auroc = roc_auc_score(gt_px, pr_px) if gt_px.size else 0
        pixel_ap = average_precision_score(gt_px, pr_px) if gt_px.size else 0
        # pixel_aupro = cal_pro_score(masks, amaps) if gt_px.size else 0
        pixel_aupro = 0
        sample_auroc = roc_auc_score(gt_sp, pr_sp) if gt_sp.size else 0
        sample_ap = average_precision_score(gt_sp, pr_sp) if gt_sp.size else 0
        
        # F1 scores
        precisions, recalls, _ = precision_recall_curve(gt_px, pr_px)
        pixel_f1 = np.max(2 * (precisions * recalls) / (precisions + recalls + 1e-8)) if gt_px.size else 0
        precisions_sp, recalls_sp, _ = precision_recall_curve(gt_sp, pr_sp)
        sample_f1 = np.max(2 * (precisions_sp * recalls_sp) / (precisions_sp + recalls_sp + 1e-8)) if gt_sp.size else 0
        
        # Format table
        table = [
            obj,
            f"{pixel_auroc * 100:.1f}",
            f"{pixel_f1 * 100:.1f}",
            f"{pixel_ap * 100:.1f}",
            f"{pixel_aupro * 100:.1f}",
            f"{sample_auroc * 100:.1f}",
            f"{sample_f1 * 100:.1f}",
            f"{sample_ap * 100:.1f}"
        ]
        table_ls.append(table)
    
    # === New: Calculate and add mean row ===
    if len(table_ls) == 0:
        return

    # Extract numeric part (skip first column class name)
    numeric_data = []
    for row in table_ls:
        numeric_values = [float(x.strip('%')) for x in row[1:]]  # Remove possible % and convert to float
        numeric_data.append(numeric_values)

    # Calculate mean for each column
    mean_values = np.array(numeric_data).mean(axis=0)
    mean_values = [f"{v:.1f}" for v in mean_values]

    # Add mean row
    mean_row = ['Mean'] + mean_values
    table_ls.append(mean_row)

    # === Generate table ===
    headers = ['Class', 'Pixel-AUROC', 'Pixel-F1', 'Pixel-AP', 'Pixel-AUPRO', 
              'Sample-AUROC', 'Sample-F1', 'Sample-AP']
    results_table = tabulate(table_ls, headers=headers, tablefmt='pipe')
    logger.info("\n%s", results_table)