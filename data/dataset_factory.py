from .datasets.nih_dataset import get_nih_dataloaders

def get_dataloaders(dataset_name, data_dir="D:/Xray_Dataset/data", batch_size=32, num_workers=4):
    """
    Factory function to return train, val, test dataloaders dynamically.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == 'nih':
        return get_nih_dataloaders(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported yet.")
