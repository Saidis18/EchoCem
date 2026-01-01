# AI generated
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

import torch

from u_net import UNet, DiceCELoss


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


DATA_DIR = Path(__file__).parent / "data"
X_TEST_DIR = DATA_DIR / "X_test_xNbnvIa" / "images"
RUNS_DIR = Path(__file__).parent / "runs"
MODEL_PATH = RUNS_DIR / "unet_model.pt"

size_labels = 272


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=3, features=[64, 128, 256], loss_fn=DiceCELoss()).to(device)
model.eval()

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing model weights: {MODEL_PATH}. Train first to create unet_model.pt."
    )

state = torch.load(MODEL_PATH, map_location=device)
if isinstance(state, dict):
    state = _strip_module_prefix(state) # type: ignore
model.load_state_dict(state, strict=False)


predictions: Dict[str, Dict[str, np.ndarray]] = {"test": {}}

for img_path in sorted(X_TEST_DIR.glob("*.npy")):
    print(f"Processing {img_path.name}...")
    name = img_path.stem
    image = np.load(img_path)

    with torch.no_grad():
        pred = model.predict(image, device).squeeze(0).numpy()
        print(f"pred shape: {pred.shape}")

    if pred.shape[1] != size_labels:
        prediction_aux = -1 + np.zeros(160 * size_labels)
        pred_flat = pred.flatten()
        num_chunks = 160
        for i in range(num_chunks):
            prediction_aux[i * size_labels : i * size_labels + 160] = pred_flat[i * 160 : (i + 1) * 160]
    else:
        prediction_aux = pred.flatten()

    predictions["test"][name] = prediction_aux

pd.DataFrame(predictions["test"], dtype="int").T.to_csv(RUNS_DIR / "y_test_csv_file.csv")
print(f"Saved predictions to {RUNS_DIR / 'y_test_csv_file.csv'}")