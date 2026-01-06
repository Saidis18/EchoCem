import loss
import pathlib
import torch


class Config():
    DATA_DIR = pathlib.Path(__file__).parent / "data"
    RUNS_DIR = pathlib.Path(__file__).parent / "runs"
    X_TEST_DIR = DATA_DIR / "X_test_xNbnvIa" / "images"
    TEST_RATIO = 0.2
    
    def __init__(
            self,
            loss_fn: torch.nn.Module,
            features: list[int],
            epochs: list[int],
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
        return sum(self.epochs)

std_configs = [
    Config(
        loss_fn=torch.nn.CrossEntropyLoss(),
        features=[64, 128, 256],
        epochs=[8, 2, 8, 2],
        batch_size_train=64,
        batch_size_val=128
    ),
    Config(
        loss_fn=loss.TVCELoss(),
        features=[64, 128, 256],
        epochs=[8, 2, 8, 2],
        batch_size_train=64,
        batch_size_val=128
    ),
    Config(
        loss_fn=loss.DiceCELoss(),
        features=[64, 128, 256],
        epochs=[8, 2, 8, 2],
        batch_size_train=64,
        batch_size_val=128
    )
]
