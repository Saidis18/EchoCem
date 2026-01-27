# Partially AI generated code
import numpy as np
import pandas as pd
from typing import Dict
import config
import torch
from segmentation import Segmentation, UNet
from weights_loader import WeightsLoader


class Benchmark:
    def __init__(self, run_num: int):
        self.run_num = run_num
        try:
            self.conf = config.std_configs[run_num - 1]
        except IndexError:
            raise ValueError(f"Invalid RUN_NUM: {run_num}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._get_model()
        self._load_weights()
    
    def _get_model(self) -> Segmentation:
        base_model = UNet(in_channels=1, out_channels=3, features=self.conf.features)
        model = Segmentation(base_model=base_model, conf=self.conf).to(self.device)
        model.eval()
        return model
    
    def _load_weights(self) -> None:
        MODEL_PATH = self.conf.RUNS_DIR / f"unet_model_{self.run_num}.pt"
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model weights: {MODEL_PATH}. Train first to create unet_model_{self.run_num}.pt.")
        WeightsLoader.load_weights(self.model, MODEL_PATH, self.device)

    def predict(self, image: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            pred = self.model.predict(image, self.device).squeeze(0).cpu().numpy()
        return pred
    
    @staticmethod
    def flatten(pred: np.ndarray) -> np.ndarray:
        size_labels = 272
        if pred.shape[1] != size_labels:
            print(f"Flattening from shape {pred.shape} to (160*{size_labels},)")
            pred_flat = np.zeros(160 * size_labels)
            pred_raw = pred.flatten()
            num_chunks = 160
            for i in range(num_chunks):
                pred_flat[i * size_labels : i * size_labels + 160] = pred_raw[i * 160 : (i + 1) * 160]
        else:
            pred_flat = pred.flatten()
        return pred_flat
    
    def run(self) -> None:
        predictions: Dict[str, Dict[str, np.ndarray]] = {"test": {}}
        for img_path in sorted(self.conf.X_TEST_DIR.glob("*.npy")):
            print(f"Processing {img_path.name}")
            name = img_path.stem
            image = np.load(img_path)
            pred = self.predict(image)
            pred_flat = self.flatten(pred)
            predictions["test"][name] = pred_flat
        y_test_dir = self.conf.RUNS_DIR / f"y_test_{self.run_num}.csv"
        pd.DataFrame(predictions["test"], dtype="int").T.to_csv(y_test_dir)
        print(f"Saved predictions to {y_test_dir}")


if __name__ == "__main__":
    RUN_NUM = 4
    benchmark = Benchmark(run_num=RUN_NUM)
    benchmark.run()
