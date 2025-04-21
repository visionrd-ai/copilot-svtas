import os
import re
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import editdistance


def merge_gt_pred(root_folder):
    """
    Merge branch and action labels into joint labels per sequence.
    Returns two dicts mapping sequence keys to merged GT and merged Pred lists.
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

            with open(agt) as f:
                action_gt = [l.strip() for l in f]
            with open(apred) as f:
                action_pred = [l.strip() for l in f]
            with open(bgt) as f:
                branch_gt = [l.strip() for l in f]
            with open(bpred) as f:
                branch_pred = [l.strip() for l in f]

            merged_gt[key] = [b + a for b, a in zip(branch_gt, action_gt)]
            merged_pred[key] = [b + a for b, a in zip(branch_pred, action_pred)]

    return merged_gt, merged_pred


def load_headwise_data(root_folder):
    """
    Load separate branch and action GT and predictions.
    Returns a dict mapping sequence keys to their headwise data.
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

            with open(agt) as f:
                action_gt = [l.strip() for l in f]
            with open(apred) as f:
                action_pred = [l.strip() for l in f]
            with open(bgt) as f:
                branch_gt = [l.strip() for l in f]
            with open(bpred) as f:
                branch_pred = [l.strip() for l in f]

            data[key] = {
                'action_gt': action_gt,
                'action_pred': action_pred,
                'branch_gt': branch_gt,
                'branch_pred': branch_pred
            }
    return data


def compute_joint_metrics(merged_gt, merged_pred):
    """
    Compute frame-level joint-label accuracy, precision, recall, F1, and normalized edit distance.
    """
    metrics = { 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'normalized_edit_distance': [] }
    for key, y_true in merged_gt.items():
        y_pred = merged_pred.get(key, [])
        if len(y_true) != len(y_pred):
            logging.warning(f"Skipping {key} (length mismatch)")
            continue
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        ed = editdistance.eval(y_true, y_pred)
        norm_ed = ed / len(y_true)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        metrics['normalized_edit_distance'].append(norm_ed)
        logging.info(f"Joint metrics for {key}: Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, NED={norm_ed:.4f}")
    avg = {m: sum(v)/len(v) if v else 0.0 for m, v in metrics.items()}
    logging.info(f"Average joint metrics: {avg}")
    return avg


def compute_headwise_metrics(data):
    """
    Compute accuracy, precision, recall, and F1 separately for branch and action heads.
    """
    head_metrics = {
        'branch': { 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [] },
        'action': { 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [] }
    }
    for key, d in data.items():
        for head in ('branch', 'action'):
            y_true = d[f'{head}_gt']
            y_pred = d[f'{head}_pred']
            if len(y_true) != len(y_pred):
                logging.warning(f"Skipping {key} {head} (length mismatch)")
                continue
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            head_metrics[head]['accuracy'].append(acc)
            head_metrics[head]['precision'].append(precision)
            head_metrics[head]['recall'].append(recall)
            head_metrics[head]['f1_score'].append(f1)
            logging.info(f"{head.capitalize()} metrics for {key}: Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
    avg = {
        head: { m: sum(vals)/len(vals) if vals else 0.0 for m, vals in metrics.items() }
        for head, metrics in head_metrics.items()
    }
    logging.info(f"Average headwise metrics: {avg}")
    return avg


def compute_per_class_joint_report(merged_gt, merged_pred):
    """
    Generate per-class precision/recall/F1 for joint labels.
    """
    y_true_all, y_pred_all = [], []
    for key in merged_gt:
        y_true_all.extend(merged_gt[key])
        y_pred_all.extend(merged_pred.get(key, []))
    report = classification_report(y_true_all, y_pred_all, zero_division=0, output_dict=True)
    logging.info(f"Per-class report for joint labels:\n{classification_report(y_true_all, y_pred_all, zero_division=0)}")
    return report


def compute_per_class_head_reports(data):
    """
    Generate per-class precision/recall/F1 separately for branch and action heads.
    """
    branch_true, branch_pred = [], []
    action_true, action_pred = [], []
    for d in data.values():
        branch_true.extend(d['branch_gt'])
        branch_pred.extend(d['branch_pred'])
        action_true.extend(d['action_gt'])
        action_pred.extend(d['action_pred'])
    branch_report = classification_report(branch_true, branch_pred, zero_division=0, output_dict=True)
    action_report = classification_report(action_true, action_pred, zero_division=0, output_dict=True)
    logging.info(f"Per-class report for branch head:\n{classification_report(branch_true, branch_pred, zero_division=0)}")
    logging.info(f"Per-class report for action head:\n{classification_report(action_true, action_pred, zero_division=0)}")
    return {'branch_report': branch_report, 'action_report': action_report}


def get_all_metrics(folder_path, log_file='metrics.log'):
    """
    Compute and log joint metrics, headwise metrics, and per-class reports for both joint labels and individual heads.
    Results are written to the specified log_file.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Starting metrics computation for folder: {folder_path}")

    merged_gt, merged_pred = merge_gt_pred(folder_path)
    data = load_headwise_data(folder_path)

    joint = compute_joint_metrics(merged_gt, merged_pred)
    headwise = compute_headwise_metrics(data)
    joint_report = compute_per_class_joint_report(merged_gt, merged_pred)
    head_reports = compute_per_class_head_reports(data)

    logging.info("Completed metrics computation.")
    logging.info(f"Summary: joint={joint}, headwise={headwise}")

    return {
        'joint': joint,
        'headwise': headwise,
        'per_class_joint': joint_report,
        'per_class_head': head_reports
    }


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('folder', help='Root folder with actions/ and branches/')
    p.add_argument('--log_file', default='metrics.log', help='Output log file path')
    args = p.parse_args()
    get_all_metrics(args.folder, args.log_file)

    print(f"Metrics written to {args.log_file}")
