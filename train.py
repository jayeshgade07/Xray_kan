import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import csv

from data.dataset_factory import get_dataloaders
from models.cnn_kan_model import CNNBaseline, CNNDense, CNNKAN
from utils.logger import setup_logger

def train_model(args):
    os.makedirs('results/logs', exist_ok=True)
    logger = setup_logger('train', f'results/logs/train_{args.model_type}.log')
    logger.info(f"Starting Training: Model={args.model_type}, Dataset={args.dataset}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Dataloaders
    train_loader, val_loader, _ = get_dataloaders(args.dataset, data_dir=args.data_dir, batch_size=args.batch_size)
    if train_loader is None:
        logger.error("Dataloader returned None. Did you run the preprocess script?")
        return

    # 2. Model Selection
    if args.model_type == 'cnn':
        model = CNNBaseline(num_classes=14)
    elif args.model_type == 'dense':
        model = CNNDense(num_classes=14)
    elif args.model_type == 'kan':
        model = CNNKAN(num_classes=14)
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        return
        
    model = model.to(device)
    
    # 3. Setup Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3.5 Checkpoint Resuming logic
    latest_ckpt_path = f"results/checkpoints/latest_{args.model_type}_{args.dataset}.pth"
    start_epoch = 0
    start_batch = 0
    best_loss = float('inf')
    
    if os.path.exists(latest_ckpt_path):
        logger.info(f"Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch_idx']
        best_loss = checkpoint.get('best_loss', float('inf'))
    
    # Setup Metrics CSV
    os.makedirs('results/metrics', exist_ok=True)
    metrics_path = f'results/metrics/{args.model_type}_training_stats.csv'
    
    # If resuming, append. Else write header.
    file_mode = 'a' if start_epoch > 0 else 'w'
    with open(metrics_path, file_mode, newline='') as f:
        writer = csv.writer(f)
        if file_mode == 'w':
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])
    
    # 4. Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for i, (images, labels) in enumerate(pbar):
            # Skip batches if we are resuming mid-epoch
            if epoch == start_epoch and i < start_batch:
                continue
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Save checkpoint every 10 batches (about every 20 mins on CPU)
            if (i + 1) % 10 == 0:
                os.makedirs('results/checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'batch_idx': i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss
                }, latest_ckpt_path)
            
            if args.dry_run and i >= 2: break
            
        # Reset start_batch for the next epoch
        start_batch = 0
            
        # Calculate real training loss for the batches we actually processed
        processed_batches = len(train_loader) if epoch > start_epoch else (len(train_loader) - start_batch)
        if processed_batches > 0:
            train_loss /= processed_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(val_loader, desc="[Valid]", leave=False)):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                if args.dry_run and i >= 2: break
                
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss])
        
        # Checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs('results/checkpoints', exist_ok=True)
            ckpt_path = f"results/checkpoints/best_{args.model_type}_{args.dataset}.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f">>> Saved Best Model to {ckpt_path}")
            
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nih')
    parser.add_argument('--data_dir', type=str, default='./data', help="Path to the dataset directory")
    parser.add_argument('--model_type', type=str, choices=['cnn', 'dense', 'kan'], default='kan')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dry_run', action='store_true', help="Run 2 batches per epoch.")
    args = parser.parse_args()
    
    train_model(args)
