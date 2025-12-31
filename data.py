from pathlib import Path
import numpy as np
from typing import List
import torch
import torch.utils.data
from torchvision import transforms # type: ignore
from PIL import Image
import time

try:
    import polars as pl
    HAS_POLARS = True
    print("Using Polars")
except ImportError:
    import pandas as pd
    HAS_POLARS = False # type: ignore
    print("Polars not found, falling back to Pandas")


class EchoCementDataset(torch.utils.data.Dataset[torch.Tensor]):
    def __init__(self, images_dir: Path, labels_csv: Path, transform: transforms.Compose | None = None):
        self.features_dir = images_dir
        self.transform = transform
        print(f"Reading csv file from: {labels_csv}")
        init_time = time.time()
        if HAS_POLARS:
            self.labels = pl.read_csv(labels_csv) # type: ignore
            self.keys: List[str] = list(self.labels['']) # type: ignore
        else:
            self.labels = pd.read_csv(labels_csv, index_col=0) # type: ignore
            self.keys: List[str] = list(self.labels.index)
        end_time = time.time()
        print(f"Finished reading csv file in {end_time - init_time:.2f}s. Found {len(self.keys)} samples.")
    
    def __len__(self) -> int:
        return len(self.keys)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: # type: ignore
        image = np.load(self.features_dir / f"{self.keys[idx]}.npy")
        if HAS_POLARS:
            labels = self.labels.row(idx)[1:] # type: ignore
        else:
            labels = self.labels.iloc[idx] # type: ignore
        label = np.array([v for v in labels if v != -1], dtype=np.int8).reshape(160, -1) # type: ignore

        if self.transform:
            image_pil = Image.fromarray(image)
            image_out = self.transform(image_pil).to(torch.float32) # type: ignore
            label_pil = Image.fromarray(label)
            label_out = self.transform(label_pil).to(torch.long).squeeze(dim=0) # type: ignore
        else:
            image_out = torch.tensor(image, dtype=torch.float32).unsqueeze(dim=0)
            label_out = torch.tensor(label, dtype=torch.long)
        return image_out, label_out # type: ignore


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    DATA_DIR = Path(__file__).parent / "data"
    # X_DIR = DATA_DIR / "X_train_uDRk9z9" / "images"
    # Y_CSV = DATA_DIR / 'Y_train_T9NrBYo.csv'

    X_DIR = DATA_DIR / "X_test_xNbnvIa" / "images"
    Y_CSV = DATA_DIR / 'y_test_csv_file.csv'

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    dataset = EchoCementDataset(X_DIR, Y_CSV, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Dataset size: {len(dataloader)}")
    for image, label in dataloader:
        print(f"Batch image shape: {image.shape}, Batch label shape: {label.shape}")
        image = image.squeeze(0, 1)
        label = label.squeeze(0)
        print(f"Image shape: {image.shape}, Label shape: {label.shape}")
        f, axarr = plt.subplots(2, 1) # type: ignore
        axarr[0].imshow(image)
        axarr[1].imshow(label)
        plt.show() # type: ignore
