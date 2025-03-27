import re
import matplotlib.pyplot as plt

def parse_logs(log_file):
    metrics = {
        "train": {"epochs": [], "action_loss": [], "branch_loss": [], "f1_action": [], "f1_branch": [], "acc_action": [], "acc_branch": []},
        "val": {"epochs": [], "action_loss": [], "branch_loss": [], "f1_action": [], "f1_branch": [], "acc_action": [], "acc_branch": []}
    }
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            train_match = re.search(r'END epoch:(\d+)\s+.*?train.*?action_loss_avg: ([\d.]+) branch_loss_avg: ([\d.]+)\s+F1Action@0.5_avg: ([\d.]+) F1Branch@0.5_avg: ([\d.]+)\s+Acc_avg: ([\d.]+) ActionSeg_Acc_avg: ([\d.]+) Acc_avg: ([\d.]+)', line)
            val_match = re.search(
                r'END epoch:(\d+)\s+.*?val.*?'
                r'action_loss_avg: ([\d.]+) branch_loss_avg: ([\d.]+)\s+'
                r'F1Action@0.5_avg: ([\d.]+) F1Branch@0.5_avg: ([\d.]+)\s+'
                r'Acc_avg: ([\d.]+) ActionSeg_Acc_avg: ([\d.]+) Acc_avg: ([\d.]+)', 
                line
            )
            if train_match:

                epoch = int(train_match.group(1))
                metrics["train"]["epochs"].append(epoch)
                metrics["train"]["action_loss"].append(float(train_match.group(2)))
                metrics["train"]["branch_loss"].append(float(train_match.group(3)))
                metrics["train"]["f1_action"].append(float(train_match.group(4)))
                metrics["train"]["f1_branch"].append(float(train_match.group(5)))
                metrics["train"]["acc_action"].append(float(train_match.group(6)))
                metrics["train"]["acc_branch"].append(float(train_match.group(8)))
                
            if val_match:

                epoch = int(val_match.group(1))
                metrics["val"]["epochs"].append(epoch)
                metrics["val"]["action_loss"].append(float(val_match.group(2)))
                metrics["val"]["branch_loss"].append(float(val_match.group(3)))
                metrics["val"]["f1_action"].append(float(val_match.group(4)))
                metrics["val"]["f1_branch"].append(float(val_match.group(5)))
                metrics["val"]["acc_action"].append(float(val_match.group(6)))
                metrics["val"]["acc_branch"].append(float(val_match.group(8)))
    
    return metrics

def plot_metrics(metrics, dataset):
    epochs = metrics["epochs"]
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f'{dataset.capitalize()} Metrics Over Epochs', fontsize=16)
    
    axs[0, 0].plot(epochs, metrics["action_loss"], marker='o', label='Action Loss')
    axs[0, 0].set_title("Action Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    
    axs[0, 1].plot(epochs, metrics["branch_loss"], marker='o', label='Branch Loss', color='r')
    axs[0, 1].set_title("Branch Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    
    axs[1, 0].plot(epochs, metrics["f1_action"], marker='o', label='F1 Action', color='g')
    axs[1, 0].set_title("F1 Score (Action)")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("F1 Score")
    
    axs[1, 1].plot(epochs, metrics["f1_branch"], marker='o', label='F1 Branch', color='m')
    axs[1, 1].set_title("F1 Score (Branch)")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("F1 Score")
    
    axs[2, 0].plot(epochs, metrics["acc_action"], marker='o', label='Action Accuracy', color='b')
    axs[2, 0].set_title("Action Accuracy")
    axs[2, 0].set_xlabel("Epoch")
    axs[2, 0].set_ylabel("Accuracy")
    
    axs[2, 1].plot(epochs, metrics["acc_branch"], marker='o', label='Branch Accuracy', color='c')
    axs[2, 1].set_title("Branch Accuracy")
    axs[2, 1].set_xlabel("Epoch")
    axs[2, 1].set_ylabel("Accuracy")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'output/Thal_amur_data/{dataset}_metrics.png')

if __name__ == "__main__":
    log_file = "output/Thal_amur_data/streamseg2d_multilabel_Thal_32x16x2.log"  # Replace with your log file path
    metrics = parse_logs(log_file)
    
    plot_metrics(metrics["train"], "train")
    plot_metrics(metrics["val"], "val")
