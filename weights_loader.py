# Partially AI generated code
import torch
from typing import Any, Dict, List
from pathlib import Path


class WeightsLoader:
    """Independent class for loading and normalizing PyTorch model weights."""

    @staticmethod
    def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove 'module.' prefix from state dict keys (from DataParallel)."""
        if not any(k.startswith("module.") for k in state_dict.keys()):
            return state_dict
        return {k[len("module.") :]: v for k, v in state_dict.items()}

    @staticmethod
    def extract_state_dict(obj: object) -> Dict[str, torch.Tensor]:
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
    def normalize_state_dict(
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
        candidates.append(WeightsLoader.strip_module_prefix(base))
        candidates.append(replace_prefix("base_model.module.", "base_model.", base))
        candidates.append(replace_prefix("base_model.module.", "base_model.", WeightsLoader.strip_module_prefix(base)))
        candidates.append(strip_prefix("module.", base))
        candidates.append(replace_prefix("base_model.module.", "base_model.", strip_prefix("module.", base)))
        candidates.append(replace_prefix("module.base_model.", "base_model.", base))
        candidates.append(replace_prefix("module.base_model.module.", "base_model.", base))

        best = max(candidates, key=score)
        return best

    @staticmethod
    def assert_state_dict_compatible(
        loaded: Dict[str, torch.Tensor],
        expected: Dict[str, torch.Tensor],
        *,
        context: str,
    ) -> None:
        """Verify that loaded and expected state dicts are compatible."""
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

    @staticmethod
    def load_weights(
        model: torch.nn.Module,
        model_path: Path,
        device: torch.device,
    ) -> None:
        """Load weights from a checkpoint file into the model."""
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model weights: {model_path}")
        
        raw = torch.load(model_path, map_location=device)
        state_dict = WeightsLoader.extract_state_dict(raw)
        
        expected = model.state_dict()
        normalized = WeightsLoader.normalize_state_dict(state_dict, expected)
        WeightsLoader.assert_state_dict_compatible(normalized, expected, context=f"loading {model_path.name}")
        
        # Now safe to load strictly.
        model.load_state_dict(normalized, strict=True)
