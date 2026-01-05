from loss import DiceCELoss, TVCELoss
from pathlib import Path
import torch
from torchvision import transforms # type: ignore


Transformation = transforms.Compose | torch.nn.Module

class Config():
    DATA_DIR = Path(__file__).parent / "data"
    RUNS_DIR = Path(__file__).parent / "runs"
    X_TEST_DIR = DATA_DIR / "X_test_xNbnvIa" / "images"
    TEST_RATIO = 0.2
    
    def __init__(
            self,
            loss_fn: torch.nn.Module,
            features: list[int],
            trans_in: Transformation,
            trans_out: Transformation,
            epochs: int,
            batch_size_train: int,
            batch_size_val: int
        ):
        self.loss_fn = loss_fn
        self.features = features
        self.trans_in = trans_in
        self.trans_out = trans_out
        self.epochs = epochs
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

std_configs = [
    Config(
        loss_fn=torch.nn.CrossEntropyLoss(),
        features=[64, 128, 256],
        trans_in=transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 0, min(img.width, 160), img.height))), # type: ignore
            transforms.ToTensor()
        ]),
        trans_out=transforms.Pad((0, 0, 272 - 160, 0)),
        epochs=30,
        batch_size_train=64,
        batch_size_val=128
    ),
    Config(
        loss_fn=TVCELoss(),
        features=[64, 128, 256],
        trans_in=transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 0, min(img.width, 160), img.height))), # type: ignore
            transforms.ToTensor()
        ]),
        trans_out=transforms.Pad((0, 0, 272 - 160, 0)),
        epochs=15,
        batch_size_train=16,
        batch_size_val=128
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
