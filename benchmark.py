# Partially AI generated code
import numpy as np
import pandas as pd
from typing import Any, Dict, List
import config
import torch
from segmentation import Segmentation, UNet


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
    
    @staticmethod
    def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not any(k.startswith("module.") for k in state_dict.keys()):
            return state_dict
        return {k[len("module.") :]: v for k, v in state_dict.items()}

    @staticmethod
    def _extract_state_dict(obj: object) -> Dict[str, torch.Tensor]:
        """Return a plain tensor state_dict from a torch.load() payload.

        Supports:
        - raw state_dict (mapping of name -> Tensor)
        - checkpoint dicts containing common keys like 'state_dict'/'model_state_dict'
        """
        if isinstance(obj, dict):
            d_any: Dict[Any, Any] = obj  # type: ignore[assignment]
            if d_any and all(isinstance(v, torch.Tensor) for v in d_any.values()):
                return d_any  # type: ignore[return-value]

            for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
                maybe = d_any.get(key)
                if isinstance(maybe, dict):
                    maybe_any: Dict[Any, Any] = maybe  # type: ignore[assignment]
                    if maybe_any and all(isinstance(v, torch.Tensor) for v in maybe_any.values()):
                        return maybe_any  # type: ignore[return-value]

        raise TypeError(
            "Unsupported checkpoint format: expected a state_dict (name->Tensor) "
            "or a dict containing one under keys like 'state_dict' or 'model_state_dict'."
        )

    @staticmethod
    def _normalize_state_dict(
        state_dict: Dict[str, torch.Tensor],
        expected_state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Try common DataParallel prefix normalizations and pick the best match."""

        expected_keys = set(expected_state_dict.keys())

        def score(candidate: Dict[str, torch.Tensor]) -> int:
            return len(expected_keys.intersection(candidate.keys()))

        def strip_prefix(prefix: str, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            if not any(k.startswith(prefix) for k in d.keys()):
                return d
            return {k[len(prefix) :]: v for k, v in d.items() if k.startswith(prefix)}

        def replace_prefix(old: str, new: str, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            if not any(k.startswith(old) for k in d.keys()):
                return d
            return { (new + k[len(old) :]) if k.startswith(old) else k: v for k, v in d.items() }

        candidates: List[Dict[str, torch.Tensor]] = []
        base = state_dict
        candidates.append(base)
        candidates.append(Benchmark._strip_module_prefix(base))
        candidates.append(replace_prefix("base_model.module.", "base_model.", base))
        candidates.append(replace_prefix("base_model.module.", "base_model.", Benchmark._strip_module_prefix(base)))
        candidates.append(strip_prefix("module.", base))
        candidates.append(replace_prefix("base_model.module.", "base_model.", strip_prefix("module.", base)))
        candidates.append(replace_prefix("module.base_model.", "base_model.", base))
        candidates.append(replace_prefix("module.base_model.module.", "base_model.", base))

        best = max(candidates, key=score)
        return best

    @staticmethod
    def _assert_state_dict_compatible(
        loaded: Dict[str, torch.Tensor],
        expected: Dict[str, torch.Tensor],
        *,
        context: str,
    ) -> None:
        loaded_keys = set(loaded.keys())
        expected_keys = set(expected.keys())

        missing = sorted(expected_keys - loaded_keys)
        unexpected = sorted(loaded_keys - expected_keys)

        shape_mismatches: List[str] = []
        for k in sorted(expected_keys.intersection(loaded_keys)):
            try:
                if tuple(loaded[k].shape) != tuple(expected[k].shape):
                    shape_mismatches.append(
                        f"{k}: expected {tuple(expected[k].shape)} got {tuple(loaded[k].shape)}"
                    )
            except Exception:
                continue

        if missing or unexpected or shape_mismatches:
            parts: List[str] = [f"State dict mismatch ({context})."]
            if missing:
                parts.append(f"Missing keys ({len(missing)}): {missing[:20]}" + (" ..." if len(missing) > 20 else ""))
            if unexpected:
                parts.append(
                    f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}" + (" ..." if len(unexpected) > 20 else "")
                )
            if shape_mismatches:
                parts.append(
                    f"Shape mismatches ({len(shape_mismatches)}): {shape_mismatches[:10]}" + (" ..." if len(shape_mismatches) > 10 else "")
                )
            raise RuntimeError("\n".join(parts))
    
    def _get_model(self) -> Segmentation:
        base_model = UNet(in_channels=1, out_channels=3, features=self.conf.features)
        model = Segmentation(base_model=base_model, conf=self.conf).to(self.device)
        model.eval()
        return model
    
    def _load_weights(self) -> None:
        MODEL_PATH = self.conf.RUNS_DIR / f"unet_model_{self.run_num}.pt"
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model weights: {MODEL_PATH}. Train first to create unet_model_{self.run_num}.pt.")
        raw = torch.load(MODEL_PATH, map_location=self.device)
        state_dict = self._extract_state_dict(raw)

        expected = self.model.state_dict()
        normalized = self._normalize_state_dict(state_dict, expected)
        self._assert_state_dict_compatible(normalized, expected, context=f"loading {MODEL_PATH.name}")

        # Now safe to load strictly.
        self.model.load_state_dict(normalized, strict=True)

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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    RUN_NUM = 1  # Change this to select different configurations
    benchmark = Benchmark(run_num=RUN_NUM)
    benchmark.run()
