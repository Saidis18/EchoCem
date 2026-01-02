from u_net import UNet, DiceCELoss
import data
import torch
import torch.utils.data # type: ignore
from torchvision import transforms # type: ignore
from torch.utils.data import DataLoader


RUN_NUM = 3
TESTING = True

if RUN_NUM == 1: # type: ignore
    loss = torch.nn.CrossEntropyLoss()
    features = [64, 128, 256, 512]
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.crop((0, 0, min(img.width, 160), img.height))), # type: ignore
        transforms.ToTensor()
    ])
    epochs = 30
    batch_size_train = 128
    batch_size_val = 64
elif RUN_NUM == 2: # type: ignore
    loss = torch.nn.CrossEntropyLoss()
    features = [64, 128, 256]
    transform = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
    epochs = 30
    batch_size_train = 128
    batch_size_val = 64
elif RUN_NUM == 3: # type: ignore
    loss = DiceCELoss()
    features = [64, 128, 256]
    transform = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
    epochs = 30
    batch_size_train = 128
    batch_size_val = 64
else:
    raise ValueError(f"Invalid RUN_NUM: {RUN_NUM}")


if TESTING:
    epochs = 2
    batch_size_train = 4
    batch_size_val = 4

if __name__ == "__main__":
    # Dataset
    DATA_DIR = data.Path(__file__).parent / "data"
    X_DIR = DATA_DIR / "X_train_uDRk9z9" / "images"
    Y_CSV = DATA_DIR / 'Y_train_T9NrBYo.csv'

    dataset = data.EchoCementDataset(X_DIR, Y_CSV, transform=transform)
    if TESTING:
        dataset = torch.utils.data.Subset(dataset, list(range(64)))
    
    train_dataset, val_dataset = data.train_test_split(dataset, test_ratio=0.2)
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=6)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=3, features=features, loss_fn=loss).to(device)
    print(f"Trainable parameters: {model.param_count}")
    print(f"Using device: {device}")

    model.training_loop(train_loader, val_loader, epochs=epochs, device=device)

    torch.save(model.state_dict(), f"runs/unet_model_{RUN_NUM}.pt")
