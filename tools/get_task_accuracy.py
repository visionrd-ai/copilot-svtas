import os
import re
def merge_gt_pred(root_folder):
    actions_folder = os.path.join(root_folder, 'actions')
    branches_folder = os.path.join(root_folder, 'branches')

    merged_gt = {}
    merged_pred = {}
    filenames = os.listdir(actions_folder)
    filtered = [f for f in filenames if re.fullmatch(r"\d+_(gt|pred)\.txt", f)]
    for filename in filtered:
        if filename.endswith('_gt.txt'):
            base_name = filename.replace('_gt.txt', '')
            action_gt_path = os.path.join(actions_folder, f'{base_name}_gt.txt')
            action_pred_path = os.path.join(actions_folder, f'{base_name}_pred.txt')

            branch_gt_path = os.path.join(branches_folder, f'{base_name}_gt.txt')
            branch_pred_path = os.path.join(branches_folder, f'{base_name}_pred.txt')

            # Read all lines from each file
            with open(action_gt_path, 'r') as f:
                action_gt = [line.strip() for line in f]

            with open(branch_gt_path, 'r') as f:
                branch_gt = [line.strip() for line in f]

            with open(action_pred_path, 'r') as f:
                action_pred = [line.strip() for line in f]

            with open(branch_pred_path, 'r') as f:
                branch_pred = [line.strip() for line in f]

            # Merge corresponding lines
            merged_gt[base_name] = [b + a for b, a in zip(branch_gt, action_gt)]
            merged_pred[base_name] = [b + a for b, a in zip(branch_pred, action_pred)]

    return merged_gt, merged_pred

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import editdistance

def compute_dataset_metrics(merged_gt, merged_pred):
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'normalized_edit_distance': []
    }

    for key in merged_gt:
        y_true = merged_gt[key]
        y_pred = merged_pred[key]

        if len(y_true) != len(y_pred):
            print(f"Skipping {key} due to length mismatch.")
            continue

        acc = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        ed = editdistance.eval(y_true, y_pred)
        norm_ed = ed / len(y_true)

        all_metrics['accuracy'].append(acc)
        all_metrics['precision'].append(precision)
        all_metrics['recall'].append(recall)
        all_metrics['f1_score'].append(f1)
        all_metrics['normalized_edit_distance'].append(norm_ed)

    avg_metrics = {k: sum(v)/len(v) if v else 0.0 for k, v in all_metrics.items()}
    return avg_metrics




merged_gt, merged_pred = merge_gt_pred('output/results/thal_production')
metrics = compute_dataset_metrics(merged_gt, merged_pred)


import pdb; pdb.set_trace()