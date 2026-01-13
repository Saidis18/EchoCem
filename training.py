from segmentation import Segmentation, UNet
import data
import torch
import torch.utils.data # type: ignore
import config
import copy


RUN_NUM = 2
TESTING = True
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
    if PRE_TRAIN:
        print(f"Starting pre-training...")
        pretrain_conf = copy.deepcopy(conf)
        pretrain_conf.loss_fn = torch.nn.MSELoss()
        dataset = data.PreTrainingDataset(pretrain_conf.PRETRAINING_PATHS)
        data_handler = data.DataHandler(dataset, conf=pretrain_conf, testing=TESTING)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = UNet(in_channels=1, out_channels=1, features=pretrain_conf.features)
        model = Segmentation(base_model=base_model, conf=pretrain_conf).to(device)
        print(f"Trainable parameters: {model.param_count}")
        print(f"Loss function: {pretrain_conf.loss_fn}")
        print(f"Using device: {device}")

        train_loader, val_loader = data_handler.get_loaders()
        model.training_loop(train_loader, val_loader, epochs=pretrain_conf.epochs, device=device)
        torch.save(model.state_dict(), pretrain_conf.RUNS_DIR / f"unet_model_{RUN_NUM}.pt")
        print(f"Model saved to {pretrain_conf.RUNS_DIR / f'unet_model_{RUN_NUM}.pt'}")
    
    dataset = data.EchoCementDataset(conf.DATA_DIR / "X_train_uDRk9z9" / "images", conf.DATA_DIR / "Y_train_T9NrBYo.csv")
    data_handler = data.DataHandler(dataset, conf=conf, testing=TESTING)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = UNet(in_channels=1, out_channels=3, features=conf.features)
    model = Segmentation(base_model=base_model, conf=conf).to(device)
    print(f"Trainable parameters: {model.param_count}")
    print(f"Loss function: {conf.loss_fn}")
    print(f"Using device: {device}")

    train_loader, val_loader = data_handler.get_loaders()
    model.training_loop(train_loader, val_loader, epochs=conf.epochs, device=device)

    torch.save(model.state_dict(), conf.RUNS_DIR / f"unet_model_{RUN_NUM}.pt")
    print(f"Model saved to {conf.RUNS_DIR / f'unet_model_{RUN_NUM}.pt'}")
