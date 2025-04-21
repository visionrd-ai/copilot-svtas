import os
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import editdistance


def merge_gt_pred(root_folder):
    """
    Merge branch and action labels into joint labels per sequence.
    """
    actions_folder = os.path.join(root_folder, 'actions')
    branches_folder = os.path.join(root_folder, 'branches')

    merged_gt, merged_pred = {}, {}
    filenames = os.listdir(actions_folder)
    filtered = [f for f in filenames if re.fullmatch(r"\d+_(gt|pred)\.txt", f)]

    for filename in filtered:
        if filename.endswith('_gt.txt'):
            key = filename.replace('_gt.txt', '')
            agt = os.path.join(actions_folder, f'{key}_gt.txt')
            apred = os.path.join(actions_folder, f'{key}_pred.txt')
            bgt = os.path.join(branches_folder, f'{key}_gt.txt')
            bpred = os.path.join(branches_folder, f'{key}_pred.txt')

            with open(agt) as f: action_gt = [l.strip() for l in f]
            with open(apred) as f: action_pred = [l.strip() for l in f]
            with open(bgt) as f: branch_gt = [l.strip() for l in f]
            with open(bpred) as f: branch_pred = [l.strip() for l in f]

            merged_gt[key] = [b + a for b, a in zip(branch_gt, action_gt)]
            merged_pred[key] = [b + a for b, a in zip(branch_pred, action_pred)]

    return merged_gt, merged_pred


def load_headwise_data(root_folder):
    """
    Load separate branch and action GT and predictions.
    """
    actions_folder = os.path.join(root_folder, 'actions')
    branches_folder = os.path.join(root_folder, 'branches')
    data = {}

    filenames = os.listdir(actions_folder)
    filtered = [f for f in filenames if re.fullmatch(r"\d+_(gt|pred)\.txt", f)]
    for filename in filtered:
        if filename.endswith('_gt.txt'):
            key = filename.replace('_gt.txt', '')
            agt = os.path.join(actions_folder, f'{key}_gt.txt')
            apred = os.path.join(actions_folder, f'{key}_pred.txt')
            bgt = os.path.join(branches_folder, f'{key}_gt.txt')
            bpred = os.path.join(branches_folder, f'{key}_pred.txt')

            with open(agt) as f: action_gt = [l.strip() for l in f]
            with open(apred) as f: action_pred = [l.strip() for l in f]
            with open(bgt) as f: branch_gt = [l.strip() for l in f]
            with open(bpred) as f: branch_pred = [l.strip() for l in f]

            data[key] = {
                'action_gt': action_gt,
                'action_pred': action_pred,
                'branch_gt': branch_gt,
                'branch_pred': branch_pred
            }
    return data


def compute_joint_metrics(merged_gt, merged_pred):
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'normalized_edit_distance': []}
    for key, y_true in merged_gt.items():
        y_pred = merged_pred[key]
        if len(y_true) != len(y_pred): continue
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        ed = editdistance.eval(y_true, y_pred)
        norm_ed = ed / len(y_true)
        for m, val in zip(['accuracy', 'precision', 'recall', 'f1_score', 'normalized_edit_distance'],
                          [acc, precision, recall, f1, norm_ed]):
            metrics[m].append(val)
        print(f"\nJoint metrics for {key}: Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, NED={norm_ed:.4f}")
    avg = {m: sum(vals)/len(vals) if vals else 0.0 for m, vals in metrics.items()}
    print(f"\nAverage joint metrics: {avg}")
    return avg


def compute_headwise_metrics(data):
    head_metrics = {
        'branch': {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []},
        'action': {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    }
    for key, d in data.items():
        for head in ('branch', 'action'):
            y_true = d[f'{head}_gt']; y_pred = d[f'{head}_pred']
            if len(y_true) != len(y_pred): continue
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            for m, val in zip(['accuracy', 'precision', 'recall', 'f1_score'], [acc, precision, recall, f1]):
                head_metrics[head][m].append(val)
            print(f"\n{head.capitalize()} metrics for {key}: Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
    avg = {head: {m: sum(vals)/len(vals) if vals else 0.0 for m, vals in metrics.items()}
           for head, metrics in head_metrics.items()}
    print(f"\nAverage headwise metrics: {avg}")
    return avg


def compute_per_class_head_reports(data):
    """
    Generate per-class reports separately for branch and action heads.
    """
    branch_true, branch_pred = [], []
    action_true, action_pred = [], []
    for d in data.values():
        branch_true.extend(d['branch_gt']); branch_pred.extend(d['branch_pred'])
        action_true.extend(d['action_gt']); action_pred.extend(d['action_pred'])
    print("\nPer-class report for branch head:\n", classification_report(branch_true, branch_pred, zero_division=0))
    print("\nPer-class report for action head:\n", classification_report(action_true, action_pred, zero_division=0))
    return {
        'branch_report': classification_report(branch_true, branch_pred, zero_division=0, output_dict=True),
        'action_report': classification_report(action_true, action_pred, zero_division=0, output_dict=True)
    }


def get_all_metrics(folder_path):
    merged_gt, merged_pred = merge_gt_pred(folder_path)
    data = load_headwise_data(folder_path)
    print("=== Joint metrics ==="); joint = compute_joint_metrics(merged_gt, merged_pred)
    print("\n=== Headwise metrics ==="); headwise = compute_headwise_metrics(data)
    print("\n=== Per-class headwise reports ==="); per_class = compute_per_class_head_reports(data)
    return {'joint': joint, 'headwise': headwise, 'per_class': per_class}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('folder', help='Root folder with actions/ and branches/')
    args = p.parse_args()
    get_all_metrics(args.folder)
