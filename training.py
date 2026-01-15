from segmentation import Segmentation, UNet
import data
import torch
import torch.utils.data # type: ignore
import config
from weights_loader import WeightsLoader
from pathlib import Path


RUN_NUM = 2
TESTING = False
PRE_TRAIN = True

try:
    conf = config.std_configs[RUN_NUM-1]
except IndexError:
    raise ValueError(f"Invalid RUN_NUM: {RUN_NUM}")

if TESTING:
    conf.epochs = 2
    conf.batch_size_train = 4
    conf.batch_size_val = 4


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = UNet(in_channels=1, out_channels=3, features=conf.features)
    model = Segmentation(base_model=base_model, conf=conf).to(device)

    pretrained_path = conf.RUNS_DIR / f"pretrained_unet_{RUN_NUM}.pt"

    if PRE_TRAIN and not pretrained_path.exists():
        print(f"Starting pre-training...")
        dataset = data.PreTrainingDataset(conf.PRETRAINING_PATHS)
        data_handler = data.DataHandler(dataset, conf=conf, testing=TESTING)
        
        base_model = UNet(in_channels=1, out_channels=1, features=conf.features)
        pretrained_model = Segmentation(base_model=base_model, conf=conf).to(device)
        pretrained_model.loss_fn = torch.nn.L1Loss()
        print(f"Trainable parameters: {pretrained_model.param_count}")
        print(f"Loss function: {pretrained_model.loss_fn}")
        print(f"Using device: {device}")

        train_loader, val_loader = data_handler.get_loaders()
        pretrained_model.training_loop(train_loader, val_loader, epochs=conf.epochs, device=device)
        # Save pretrained model
        torch.save(pretrained_model.state_dict(), pretrained_path)
        print(f"Pretrained model saved to {pretrained_path}")
        
        # Load encoder and decoder weights from pretrained model (strict=False allows final_conv mismatch)
        WeightsLoader.load_weights(model.base_model, Path(pretrained_path), device, strict=False)
        print(f"Loaded encoder/decoder weights from {pretrained_path} (final_conv will be randomly initialized for 3-class output)")
        print(f"Pre-training completed. Proceeding to segmentation training...")
    
    dataset = data.EchoCementDataset(conf.DATA_DIR / "X_train_uDRk9z9" / "images", conf.DATA_DIR / "Y_train_T9NrBYo.csv")
    data_handler = data.DataHandler(dataset, conf=conf, testing=TESTING)
    

    print(f"Trainable parameters: {model.param_count}")
    print(f"Loss function: {model.loss_fn}")
    print(f"Using device: {device}")

    train_loader, val_loader = data_handler.get_loaders()
    model.training_loop(train_loader, val_loader, epochs=conf.epochs, device=device)

    torch.save(model.state_dict(), conf.RUNS_DIR / f"unet_model_{RUN_NUM}.pt")
    print(f"Model saved to {conf.RUNS_DIR / f'unet_model_{RUN_NUM}.pt'}")
