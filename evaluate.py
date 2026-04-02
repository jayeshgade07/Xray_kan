import argparse
import torch
import torch.nn as nn
from data.dataset_factory import get_dataloaders
from models.cnn_kan_model import CNNBaseline, CNNDense, CNNKAN
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, auc
from data.datasets.nih_dataset import NIH_DISEASES
import numpy as np
from tqdm import tqdm
import os
import csv
from utils.logger import setup_logger

def evaluate_model(args):
    os.makedirs('results/logs', exist_ok=True)
    logger = setup_logger('eval', f'results/logs/eval_{args.model_type}.log')
    logger.info(f"Starting Evaluation: Model={args.model_type}, Dataset={args.dataset}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    _, _, test_loader = get_dataloaders(args.dataset, data_dir=args.data_dir, batch_size=args.batch_size)
    if test_loader is None:
        logger.error("Dataloader returned None.")
        return
        
    if args.model_type == 'cnn':
        model = CNNBaseline(num_classes=14)
    elif args.model_type == 'dense':
        model = CNNDense(num_classes=14)
    elif args.model_type == 'kan':
        model = CNNKAN(num_classes=14)
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        return
        
    checkpoint_path = f"results/checkpoints/best_{args.model_type}_{args.dataset}.pth"
    if args.load_ckpt:
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}. Running with random weights.")
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            outputs = torch.sigmoid(model(images)).cpu().numpy()
            all_preds.append(outputs)
            all_labels.append(labels.numpy())
            if args.dry_run and i >= 2: break
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total Parameters: {total_params:,}")
    
    try:
        macro_auc = roc_auc_score(all_labels, all_preds, average='macro')
        pred_bin = (all_preds > 0.5).astype(int)
        f1 = f1_score(all_labels, pred_bin, average='macro')
        acc = accuracy_score(all_labels, pred_bin)
        
        logger.info(f"Macro AUC:        {macro_auc:.4f}")
        logger.info(f"Macro F1 Score:   {f1:.4f}")
        logger.info(f"Exact Match Acc:  {acc:.4f}")
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        for i in range(len(NIH_DISEASES)):
            fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
            roc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{NIH_DISEASES[i]} (AUC = {roc_val:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multi-label ROC Curves - {args.model_type.upper()}')
        plt.legend(loc="lower right", fontsize='small', ncol=2)
        plt.tight_layout()
        
        os.makedirs('results/plots', exist_ok=True)
        plot_path = f'results/plots/roc_{args.model_type}.png'
        plt.savefig(plot_path)
        logger.info(f"Saved ROC curves to {plot_path}")
        
        os.makedirs('results/metrics', exist_ok=True)
        metrics_file = 'results/metrics/evaluation_summary.csv'
        write_header = not os.path.exists(metrics_file)
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['Model', 'Dataset', 'Params', 'AUC', 'F1', 'Accuracy'])
            writer.writerow([args.model_type, args.dataset, total_params, f"{macro_auc:.4f}", f"{f1:.4f}", f"{acc:.4f}"])
            
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        logger.warning("Metrics skipped (labels likely lack variance due to dry run).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nih')
    parser.add_argument('--data_dir', type=str, default='./data', help="Path to the dataset directory")
    parser.add_argument('--model_type', type=str, choices=['cnn', 'dense', 'kan'], default='kan')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--load_ckpt', action='store_true', help="Load best checkpoint if available")
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    evaluate_model(args)
