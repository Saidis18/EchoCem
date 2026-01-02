from segmentation import DiceCELoss
import data
import torch
import torch.utils.data # type: ignore
from torchvision import transforms # type: ignore
from dataclasses import dataclass


@dataclass
class Config:
    loss_fn: torch.nn.Module
    features: list[int]
    transform: transforms.Compose | None
    epochs: int
    batch_size_train: int
    batch_size_val: int

    DATA_DIR: data.Path = data.Path(__file__).parent / "data"
    X_TRAIN_DIR: data.Path = DATA_DIR / "X_train_uDRk9z9" / "images"
    Y_TRAIN_CSV: data.Path = DATA_DIR / 'Y_train_T9NrBYo.csv'
    X_TEST_DIR: data.Path = DATA_DIR / "X_test_xNbnvIa" / "images"
    RUNS_DIR: data.Path = data.Path(__file__).parent / "runs"


std_configs = [
    Config(
        loss_fn=torch.nn.CrossEntropyLoss(),
        features=[64, 128, 256, 512],
        transform=transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 0, min(img.width, 160), img.height))), # type: ignore
            transforms.ToTensor()
        ]),
        epochs=30,
        batch_size_train=128,
        batch_size_val=64
    ),
    Config(
        loss_fn=torch.nn.CrossEntropyLoss(),
        features=[64, 128, 256],
        transform=transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()]),
        epochs=30,
        batch_size_train=128,
        batch_size_val=64
    ),
    Config(
        loss_fn=DiceCELoss(),
        features=[64, 128, 256],
        transform=transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()]),
        epochs=30,
        batch_size_train=128,
        batch_size_val=64
    )
]
