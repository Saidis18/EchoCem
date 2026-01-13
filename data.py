from pathlib import Path
import numpy as np
from typing import List, Tuple
import torch
import torch.utils.data
import time
import config

try:
    import polars as pl
    HAS_POLARS = True
    print("Using Polars")
except ImportError:
    import pandas as pd
    HAS_POLARS = False # type: ignore
    print("Polars not found, falling back to Pandas")


BaseSubset = torch.utils.data.Subset[torch.Tensor]
class BaseDataset(torch.utils.data.Dataset[torch.Tensor]):
    def __init__(self):
        super().__init__()
    
    def __len__(self) -> int:
        raise NotImplementedError()


class EchoCementDataset(BaseDataset):
    def __init__(self, images_dir: Path, labels_csv: Path):
        self.features_dir = images_dir
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
        image_out = torch.tensor(image, dtype=torch.float32).unsqueeze(dim=0)
        label_out = torch.tensor(label, dtype=torch.long)
        size = image_out.shape[2]
        image_out = torch.nn.functional.pad(image_out, (0, 272 - size), mode='constant', value=0)
        label_out = torch.nn.functional.pad(label_out, (0, 272 - size), mode='constant', value=0)
        return image_out, label_out # type: ignore


class DataHandler():
    def __init__(self, positions: List[Tuple[Path, Path]], conf: config.Config, testing: bool = False):
        self.conf = conf
        self.loaders: List[Tuple[torch.utils.data.DataLoader[torch.Tensor], torch.utils.data.DataLoader[torch.Tensor]]] = []
        for images_dir, labels_csv in positions:
            dataset = EchoCementDataset(images_dir, labels_csv)
            if testing:
                dataset = torch.utils.data.Subset(dataset, list(range(32)))
            train_dataset, val_dataset = DataHandler.train_test_split(dataset, test_ratio=self.conf.TEST_RATIO)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.conf.batch_size_train, shuffle=True, num_workers=6)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.conf.batch_size_val, shuffle=False, num_workers=6)
            self.loaders.append((train_loader, val_loader))
        assert len(self.loaders) == max(dataloader_idx for dataloader_idx, _ in self.conf.epochs) + 1, "Number of dataloaders does not match config."
    
    def get_loaders(self):
        for dataloader_idx, epochs in self.conf.epochs:
            train_loader, val_loader = self.loaders[dataloader_idx]
            yield train_loader, val_loader, epochs
    
    @staticmethod
    def train_test_split(dataset: BaseDataset | BaseSubset, test_ratio: float) -> Tuple[BaseSubset, BaseSubset]:
        train_ratio = 1.0 - test_ratio
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_subset = torch.utils.data.Subset(dataset, train_dataset.indices)
        val_subset = torch.utils.data.Subset(dataset, val_dataset.indices)
        return train_subset, val_subset


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    DATA_DIR = Path(__file__).parent / "data"

    X_DIR = DATA_DIR / "X_test_xNbnvIa" / "images"
    Y_CSV = Path(__file__).parent / "runs" / 'y_test_1.csv'

    dataset = EchoCementDataset(X_DIR, Y_CSV)
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
