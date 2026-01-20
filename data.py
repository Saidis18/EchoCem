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
        image_out = torch.nn.functional.interpolate(image_out.unsqueeze(0), size=(160, 160), mode='bilinear', align_corners=False).squeeze(0)
        image_out = (image_out - image_out.min()) / (image_out.max() - image_out.min())
        label_out = torch.nn.functional.interpolate(label_out.unsqueeze(0).unsqueeze(0).float(), size=(160, 160), mode='nearest').long().squeeze(0).squeeze(0)
        return image_out, label_out # type: ignore


class PreTrainingDataset(BaseDataset):
    def __init__(self, paths: List[Path]):
        self.paths = paths
        # Cache all file paths once during initialization to ensure consistency
        self.all_files: List[Path] = []
        for path in paths:
            self.all_files.extend(sorted(list(path.glob("*.npy"))))
        print(f"PreTrainingDataset initialized with {len(self.all_files)} files from {len(paths)} directories.")
    
    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: # type: ignore
        if idx >= len(self.all_files):
            raise IndexError("Index out of range")
        
        image_path = self.all_files[idx]
        image = np.load(image_path)
        image_out = torch.tensor(image, dtype=torch.float32).unsqueeze(dim=0)
        image_out = torch.nn.functional.interpolate(image_out.unsqueeze(0), size=(160, 160), mode='bilinear', align_corners=False).squeeze(0)
        image_out = (image_out - image_out.min()) / (image_out.max() - image_out.min())
        
        # Create noisy version as label (image + Gaussian noise)
        noise = torch.randn_like(image_out)*0.1  # Standard deviation of 0.5
        noisy_image = (image_out + noise).clamp(0, 1)

        hole_size = 32
        x = torch.randint(0, 160 - hole_size, (1,)).item()
        y = torch.randint(0, 160 - hole_size, (1,)).item()
        noisy_image[:, x:x+hole_size, y:y+hole_size] = 0
        
        return noisy_image, image_out # type: ignore


class DataHandler():
    def __init__(self, dataset: BaseDataset | BaseSubset, conf: config.Config, testing: bool = False):
        self.conf = conf
        if testing:
            dataset = torch.utils.data.Subset(dataset, list(range(32)))
        train_dataset, val_dataset = DataHandler.train_test_split(dataset, test_ratio=self.conf.TEST_RATIO)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.conf.batch_size_train, shuffle=True, num_workers=6)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.conf.batch_size_val, shuffle=False, num_workers=6)

    
    def get_loaders(self):
        return self.train_loader, self.val_loader
    
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
    max_count = 20
    import matplotlib.pyplot as plt
    DATA_DIR = Path(__file__).parent / "data"

    X_DIR = DATA_DIR / "X_test_xNbnvIa" / "images"
    Y_CSV = Path(__file__).parent / "runs" / "y_test_4.csv"

    dataset = EchoCementDataset(X_DIR, Y_CSV)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Dataset size: {len(dataloader)}")
    counter = 0
    for image, label in dataloader:
        print(f"Batch image shape: {image.shape}, Batch label shape: {label.shape}")
        image = image.squeeze(0, 1)
        label = label.squeeze(0)
        print(f"Image shape: {image.shape}, Label shape: {label.shape}")
        f, axarr = plt.subplots(2, 1) # type: ignore
        axarr[0].imshow(image)
        axarr[1].imshow(label)
        plt.show() # type: ignore
        counter += 1
        if counter >= max_count:
            break
    
    paths = [
        DATA_DIR / "X_test_xNbnvIa" / "images",
        DATA_DIR / "X_train_uDRk9z9" / "images",
        DATA_DIR / "X_unlabeled_mtkxUlo" / "images"
    ]
    pretrain_dataset = PreTrainingDataset(paths)
    pretrain_dataloader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=1, shuffle=True)
    print(f"PreTraining Dataset size: {len(pretrain_dataloader)}")
    counter = 0
    for noisy_image, clean_image in pretrain_dataloader:
        print(f"Batch noisy image shape: {noisy_image.shape}, Batch clean image shape: {clean_image.shape}")
        noisy_image = noisy_image.squeeze(0, 1)
        clean_image = clean_image.squeeze(0, 1)
        print(f"Noisy Image shape: {noisy_image.shape}, Clean Image shape: {clean_image.shape}")
        f, axarr = plt.subplots(2, 1) # type: ignore
        axarr[0].imshow(noisy_image)
        axarr[1].imshow(clean_image)
        plt.show() # type: ignore
        counter += 1
        if counter >= max_count:
            break
