from pathlib import Path
import numpy as np
import polars as pl
import torch


class EchoCementDataset(torch.utils.data.Dataset[torch.Tensor]):
    def __init__(self, images_dir: Path, labels_csv: Path):
        self.features_dir = images_dir
        self.labels = pl.read_csv(labels_csv)
        self.keys = list(self.labels[''])
    
    def __len__(self) -> int:
        return len(self.keys)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: # type: ignore
        image = np.load(self.features_dir / f"{self.keys[idx]}.npy")[:, :160]
        label = np.array([np.array([1 if v==i else 0 for v in self.labels.row(idx) if v in [0, 1, 2]]).reshape(160, -1)[:, :160] for i in [0, 1, 2]])
        return torch.tensor(image, dtype=torch.float32).unsqueeze(dim=0), torch.tensor(label, dtype=torch.float32)


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
        label = label.squeeze(0)
        print(f"Image shape: {image.shape}, Label shape: {label.shape}")
        f, axarr = plt.subplots(2, 1) # type: ignore
        axarr[0].imshow(image)
        axarr[1].imshow(label.numpy().transpose(1, 2, 0))
        plt.show() # type: ignore
        break
