import argparse
import torch
import torch.nn as nn
from data.dataset_factory import get_dataloaders
from models.cnn_kan_model import CNNBaseline, CNNDense, CNNKAN
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
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
    
    _, _, test_loader = get_dataloaders(args.dataset, data_dir='D:/Xray_Dataset/data', batch_size=args.batch_size)
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
        auc = roc_auc_score(all_labels, all_preds, average='macro')
        pred_bin = (all_preds > 0.5).astype(int)
        f1 = f1_score(all_labels, pred_bin, average='macro')
        acc = accuracy_score(all_labels, pred_bin)
        
        logger.info(f"Macro AUC:        {auc:.4f}")
        logger.info(f"Macro F1 Score:   {f1:.4f}")
        logger.info(f"Exact Match Acc:  {acc:.4f}")
        
        os.makedirs('results/metrics', exist_ok=True)
        metrics_file = 'results/metrics/evaluation_summary.csv'
        write_header = not os.path.exists(metrics_file)
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['Model', 'Dataset', 'Params', 'AUC', 'F1', 'Accuracy'])
            writer.writerow([args.model_type, args.dataset, total_params, f"{auc:.4f}", f"{f1:.4f}", f"{acc:.4f}"])
            
    except Exception as e:
        logger.warning("Metrics skipped (labels likely lack variance due to dry run).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nih')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'dense', 'kan'], default='kan')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--load_ckpt', action='store_true', help="Load best checkpoint if available")
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    evaluate_model(args)
