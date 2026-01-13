import loss
import pathlib
import torch
from typing import List, Tuple


class Config():
    DATA_DIR = pathlib.Path(__file__).parent / "data"
    RUNS_DIR = pathlib.Path(__file__).parent / "runs"
    X_TEST_DIR = DATA_DIR / "X_test_xNbnvIa" / "images"
    TEST_RATIO = 0.2
    
    def __init__(
            self,
            loss_fn: torch.nn.Module,
            features: List[int],
            epochs: List[Tuple[int, int]], # (dataloader_idx, num_epochs for that dataloader)
            batch_size_train: int,
            batch_size_val: int
        ):
        self.loss_fn = loss_fn
        self.features = features
        self.epochs = epochs
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
    
    @property
    def num_epochs(self) -> int:
        return sum(epoch for _, epoch in self.epochs)

std_configs = [
    Config(
        loss_fn=torch.nn.CrossEntropyLoss(),
        features=[64, 128, 256, 512],
        epochs=[(0, 20)],
        batch_size_train=64,
        batch_size_val=128
    ),
    Config(
        loss_fn=torch.nn.CrossEntropyLoss(torch.tensor([0.6, 1.4, 1.2])),
        features=[64, 128, 256, 512],
        epochs=[(0, 20)],
        batch_size_train=64,
        batch_size_val=128
    ),
    Config(
        loss_fn=loss.TVCELoss(tv_weight=0.4, ce_weight=torch.tensor([0.6, 1.4, 1.2])),
        features=[64, 128, 256, 512],
        epochs=[(0, 20)],
        batch_size_train=64,
        batch_size_val=128
    ),
    Config(
        loss_fn=loss.DiceCELoss(),
        features=[64, 128, 256, 512],
        epochs=[(0, 20)],
        batch_size_train=64,
        batch_size_val=128
    )
]
