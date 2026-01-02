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
    conf.epochs = 2
    conf.batch_size_train = 4
    conf.batch_size_val = 4


if __name__ == "__main__":
    dataset = data.EchoCementDataset(conf.X_TRAIN_DIR, conf.Y_TRAIN_CSV, transform=conf.trans_in)
    if TESTING:
        dataset = torch.utils.data.Subset(dataset, list(range(64)))
    
    train_dataset, val_dataset = data.train_test_split(dataset, test_ratio=0.2)
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size_train, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=conf.batch_size_val, shuffle=False, num_workers=6)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = UNet(in_channels=1, out_channels=3, features=conf.features)
    model = Segmentation(base_model=base_model, loss_fn=conf.loss_fn).to(device)
    print(f"Trainable parameters: {model.param_count}")
    print(f"Loss function: {conf.loss_fn}")
    print(f"Using device: {device}")

    model.training_loop(train_loader, val_loader, epochs=conf.epochs, device=device)

    torch.save(model.state_dict(), conf.RUNS_DIR / f"unet_model_{RUN_NUM}.pt")
    print(f"Model saved to {conf.RUNS_DIR / f'unet_model_{RUN_NUM}.pt'}")
