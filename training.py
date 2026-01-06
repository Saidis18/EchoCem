from segmentation import Segmentation, UNet
import data
import torch
import torch.utils.data # type: ignore
from torch.utils.data import DataLoader
import config


RUN_NUM = 1
TESTING = False

try:
    conf = config.std_configs[RUN_NUM-1]
except IndexError:
    raise ValueError(f"Invalid RUN_NUM: {RUN_NUM}")

if TESTING:
    conf.epochs = [2, 2]
    conf.batch_size_train = 4
    conf.batch_size_val = 4


if __name__ == "__main__":
    dataset_160 = data.EchoCementDataset(conf.DATA_DIR / "X_train_160", conf.DATA_DIR / "Y_train_160.csv")
    dataset_272 = data.EchoCementDataset(conf.DATA_DIR / "X_train_272", conf.DATA_DIR / "Y_train_272.csv")
    if TESTING:
        dataset_160 = torch.utils.data.Subset(dataset_160, list(range(32)))
        dataset_272 = torch.utils.data.Subset(dataset_272, list(range(32)))
    
    train_dataset_160, val_dataset_160 = data.train_test_split(dataset_160, test_ratio=conf.TEST_RATIO)
    train_dataset_272, val_dataset_272 = data.train_test_split(dataset_272, test_ratio=conf.TEST_RATIO)
    print(f"Train dataset size: {len(train_dataset_160) + len(train_dataset_272)}, Validation dataset size: {len(val_dataset_160) + len(val_dataset_272)}")

    # Dataloaders
    train_loader_160 = DataLoader(train_dataset_160, batch_size=conf.batch_size_train, shuffle=True, num_workers=6)
    train_loader_272 = DataLoader(train_dataset_272, batch_size=conf.batch_size_train, shuffle=True, num_workers=6)
    val_loader_160 = DataLoader(val_dataset_160, batch_size=conf.batch_size_val, shuffle=False, num_workers=6)
    val_loader_272 = DataLoader(val_dataset_272, batch_size=conf.batch_size_val, shuffle=False, num_workers=6)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = UNet(in_channels=1, out_channels=3, features=conf.features)
    model = Segmentation(base_model=base_model, loss_fn=conf.loss_fn).to(device)
    print(f"Trainable parameters: {model.param_count}")
    print(f"Loss function: {conf.loss_fn}")
    print(f"Using device: {device}")

    model.training_loop(train_loader_160, val_loader_272, epochs=conf.epochs[0], device=device)
    model.training_loop(train_loader_272, val_loader_160, epochs=conf.epochs[1], device=device)

    torch.save(model.state_dict(), conf.RUNS_DIR / f"unet_model_{RUN_NUM}.pt")
    print(f"Model saved to {conf.RUNS_DIR / f'unet_model_{RUN_NUM}.pt'}")
