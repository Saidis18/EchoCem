from pathlib import Path
import numpy as np
from typing import List
import torch

try:
    import polars as pl
    HAS_POLARS = True
    print("Using Polars")
except ImportError:
    import pandas as pd
    HAS_POLARS = False
    print("Polars not found, falling back to Pandas")



class EchoCementDataset(torch.utils.data.Dataset[torch.Tensor]):
    def __init__(self, images_dir: Path, labels_csv: Path):
        self.features_dir = images_dir
        if HAS_POLARS:
            self.labels = pl.read_csv(labels_csv) # type: ignore
            self.keys: List[str] = list(self.labels['']) # type: ignore
        else:
            self.labels = pd.read_csv(labels_csv, index_col=0) # type: ignore
            self.keys: List[str] = list(self.labels.index)
    
    def __len__(self) -> int:
        return len(self.keys)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: # type: ignore
        image = np.load(self.features_dir / f"{self.keys[idx]}.npy")[:, :160]
        if HAS_POLARS:
            labels = self.labels.row(idx) # type: ignore
        else:
            labels = self.labels.iloc[idx] # type: ignore
        label = np.array([v for v in labels if v in [0, 1, 2]]).reshape(160, -1)[:, :160] # type: ignore
        return torch.tensor(image, dtype=torch.float32).unsqueeze(dim=0), torch.tensor(label, dtype=torch.float32).unsqueeze(dim=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    DATA_DIR = Path(__file__).parent / "data"
    X_DIR = DATA_DIR / "X_train_uDRk9z9" / "images"
    Y_CSV = DATA_DIR / 'Y_train_T9NrBYo.csv'

    dataset = EchoCementDataset(X_DIR, Y_CSV)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Dataset size: {len(dataloader)}")
    for image, label in dataloader:
        print(f"Batch image shape: {image.shape}, Batch label shape: {label.shape}")
        image = image.squeeze(0, 1)
        label = label.squeeze(0, 1)
        print(f"Image shape: {image.shape}, Label shape: {label.shape}")
        f, axarr = plt.subplots(2, 1) # type: ignore
        axarr[0].imshow(image)
        axarr[1].imshow(label)
        plt.show() # type: ignore
        break
