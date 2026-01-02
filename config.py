from loss import DiceCELoss
from pathlib import Path
import torch
from torchvision import transforms # type: ignore
from dataclasses import dataclass


Transformation = transforms.Compose | torch.nn.Module
@dataclass
class Config:
    loss_fn: torch.nn.Module
    features: list[int]
    trans_in: Transformation
    trans_out: Transformation
    epochs: int
    batch_size_train: int
    batch_size_val: int

    DATA_DIR: Path = Path(__file__).parent / "data"
    X_TRAIN_DIR: Path = DATA_DIR / "X_train_uDRk9z9" / "images"
    Y_TRAIN_CSV: Path = DATA_DIR / 'Y_train_T9NrBYo.csv'
    X_TEST_DIR: Path = DATA_DIR / "X_test_xNbnvIa" / "images"
    RUNS_DIR: Path = Path(__file__).parent / "runs"


std_configs = [
    Config(
        loss_fn=torch.nn.CrossEntropyLoss(),
        features=[64, 128, 256, 512],
        trans_in=transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 0, min(img.width, 160), img.height))), # type: ignore
            transforms.ToTensor()
        ]),
        trans_out=transforms.Pad((0, 0, 272 - 160, 0)),
        epochs=30,
        batch_size_train=128,
        batch_size_val=64
    ),
    Config(
        loss_fn=torch.nn.CrossEntropyLoss(),
        features=[64, 128, 256],
        trans_in=transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()]),
        trans_out=transforms.Resize((160, 272), interpolation=transforms.InterpolationMode.NEAREST),
        epochs=30,
        batch_size_train=128,
        batch_size_val=64
    ),
    Config(
        loss_fn=DiceCELoss(),
        features=[64, 128, 256],
        trans_in=transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()]),
        trans_out=transforms.Resize((160, 272), interpolation=transforms.InterpolationMode.NEAREST),
        epochs=30,
        batch_size_train=128,
        batch_size_val=64
    )
]
