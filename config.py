import loss
import pathlib
import torch
from typing import List


class Config():
    DATA_DIR = pathlib.Path(__file__).parent / "data"
    RUNS_DIR = pathlib.Path(__file__).parent / "runs"
    X_TEST_DIR = DATA_DIR / "X_test_xNbnvIa" / "images"
    TEST_RATIO = 0.0
    PRETRAINING_PATHS: List[pathlib.Path] = [
        DATA_DIR / "X_test_xNbnvIa" / "images",
        DATA_DIR / "X_train_uDRk9z9" / "images",
        DATA_DIR / "X_unlabeled_mtkxUlo" / "images"
    ]
    
    def __init__(
            self,
            loss_fn: torch.nn.Module,
            features: List[int],
            epochs: int,
            batch_size_train: int,
            batch_size_val: int
        ):
        self.loss_fn = loss_fn
        self.features = features
        self.epochs = epochs
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

std_configs = [
    Config(
        loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-100),
        features=[64, 128, 256, 512],
        epochs=30,
        batch_size_train=64,
        batch_size_val=128
    ),
    Config(
        loss_fn=torch.nn.CrossEntropyLoss(torch.tensor([0.6, 1.3, 1.3]), ignore_index=-100),
        features=[64, 128, 256, 512],
        epochs=30,
        batch_size_train=64,
        batch_size_val=128
    ),
    Config(
        loss_fn=loss.TVCELoss(tv_weight=0.4, ce_weight=torch.tensor([0.6, 1.3, 1.3])),
        features=[64, 128, 256, 512],
        epochs=30,
        batch_size_train=64,
        batch_size_val=128
    ),
    Config(
        loss_fn=loss.DiceCELoss(weight_ce=0.0),
        features=[64, 128, 256, 512],
        epochs=30,
        batch_size_train=64,
        batch_size_val=128
    )
]
