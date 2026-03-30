import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

NIH_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def preprocess_nih_metadata(data_dir="D:/Xray_Dataset/data"):
    # This script creates one-hot encoded CSVs for train, val, and test.
    csv_path = os.path.join(data_dir, 'Data_Entry_2017.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please download the dataset first.")
        return

    df = pd.read_csv(csv_path)
    
    # Process multi-label string like 'Atelectasis|Effusion' into binary columns
    for disease in NIH_DISEASES:
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)
        
    # Split by patient ID to prevent data leakage (same patient in train & test)
    patient_ids = df['Patient ID'].unique()
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.20, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=42) # 0.125 * 0.8 = 0.10 validation
    
    train_df = df[df['Patient ID'].isin(train_ids)]
    val_df = df[df['Patient ID'].isin(val_ids)]
    test_df = df[df['Patient ID'].isin(test_ids)]
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    print("Pre-processing complete. CSVs generated.")

if __name__ == "__main__":
    preprocess_nih_metadata()
