import torch
import numpy as np
from scipy import ndimage as ndi  # type: ignore


class DiceCELoss(torch.nn.Module):
    def __init__(self, weight_ce: float = 0.1, smooth: float = 1e-6):
        super(DiceCELoss, self).__init__() # type: ignore
        self.weight_ce = weight_ce
        self.smooth = smooth
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs_soft = torch.nn.functional.softmax(inputs, dim=1)
        
        # Create mask for valid targets (ignore -100)
        valid_mask = targets >= 0
        
        # Replace invalid targets with 0 temporarily for one_hot encoding
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        
        targets_one_hot = torch.nn.functional.one_hot(targets_safe, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Apply mask to both inputs and targets
        valid_mask_expanded = valid_mask.unsqueeze(1).float()
        inputs_soft_masked = inputs_soft * valid_mask_expanded
        targets_one_hot_masked = targets_one_hot * valid_mask_expanded
        
        intersection = torch.sum(inputs_soft_masked * targets_one_hot_masked)
        cardinality = torch.sum(inputs_soft_masked + targets_one_hot_masked)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice
        
        return dice_loss
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        total_loss = self.weight_ce * ce_loss + (1.0 - self.weight_ce) * dice_loss
        return total_loss


class TVCELoss(torch.nn.Module):
    def __init__(self, tv_weight: float, ce_weight: torch.Tensor = torch.tensor([1.0, 1.0, 1.0])):
        super(TVCELoss, self).__init__()  # type: ignore
        self.ce_loss = torch.nn.CrossEntropyLoss(ce_weight)
        self.tv_weight = tv_weight

    def tv_loss(self, x: torch.Tensor) -> torch.Tensor:
        ax1 = torch.abs(x[:, 1:, :] - x[:, :-1, :]).to(torch.float32)
        return torch.mean(ax1)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        tv_loss = self.tv_loss(inputs)
        total_loss = ce_loss + self.tv_weight * tv_loss
        return total_loss


class BoundaryLoss(torch.nn.Module):
    """Boundary loss based on signed distance maps (SciPy EDT).

    This follows the idea from:
        Kervadec et al., "Boundary loss for highly unbalanced segmentation".

    Notes:
    - This implementation is CPU-bound because SciPy's distance transform runs on CPU.
    - It is mathematically closer to the paper than edge-only surrogates.
    """
    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super(BoundaryLoss, self).__init__() # type: ignore
        self.ignore_index = ignore_index
        self.reduction = reduction

    def _signed_distance(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(bool)
        if mask.any():
            posmask = mask
            negmask = ~posmask
            dist_out: np.ndarray = ndi.distance_transform_edt(negmask)  # type: ignore[union-attr]
            dist_in: np.ndarray = ndi.distance_transform_edt(posmask)  # type: ignore[union-attr]
            signed = dist_out - dist_in
        else:
            signed = np.zeros_like(mask, dtype=np.float32)
        return signed.astype(np.float32)

    def _compute_distance_maps(
        self,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        num_classes: int,
        device: torch.device,
    ) -> torch.Tensor:
        targets_cpu = targets.detach().cpu().numpy()
        valid_cpu = (valid_mask.detach().cpu().numpy() != 0)
        batch_size, height, width = targets_cpu.shape
        dist_maps = np.zeros((batch_size, num_classes, height, width), dtype=np.float32)

        for b in range(batch_size):
            if not valid_cpu[b].any():
                continue
            for c in range(num_classes):
                class_mask = (targets_cpu[b] == c) & valid_cpu[b]
                dist_map = self._signed_distance(class_mask)
                dist_map[~valid_cpu[b]] = 0.0
                dist_maps[b, c] = dist_map

        return torch.from_numpy(dist_maps).to(device)  # type: ignore[reportUnknownMemberType]

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 4:
            raise ValueError("BoundaryLoss expects inputs of shape (B, C, H, W)")
        if targets.dim() != 3:
            raise ValueError("BoundaryLoss expects targets of shape (B, H, W)")

        num_classes = inputs.shape[1]
        probs = torch.nn.functional.softmax(inputs, dim=1)

        valid_mask = (targets != self.ignore_index) & (targets >= 0) & (targets < num_classes)
        targets_safe = targets.clone()
        targets_safe[valid_mask == 0] = 0

        with torch.no_grad():
            dist_maps = self._compute_distance_maps(targets_safe, valid_mask, num_classes, inputs.device)

        valid_mask_f = valid_mask.unsqueeze(1).to(dtype=probs.dtype)
        loss = (probs * dist_maps * valid_mask_f).sum()

        if self.reduction == "sum":
            return loss
        if self.reduction == "none":
            return probs * dist_maps * valid_mask_f

        denom = valid_mask_f.sum()
        return loss / (denom + 1e-8)


class DiceBoundCELoss(torch.nn.Module):
    def __init__(self, weight_ce: float = 0.1, weight_bound: float = 0.1, smooth: float = 1e-6, ignore_index: int = -100):
        super(DiceBoundCELoss, self).__init__() # type: ignore
        self.weight_ce = weight_ce
        self.weight_bound = weight_bound
        self.smooth = smooth
        self.dice_loss = DiceCELoss(weight_ce=weight_ce, smooth=smooth)
        self.boundary_loss = BoundaryLoss(ignore_index=ignore_index)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.dice_loss.ce_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        bound_loss = self.boundary_loss(inputs, targets)
        total_loss = (self.weight_ce * ce_loss +
                      (1.0 - self.weight_ce - self.weight_bound) * dice_loss +
                      self.weight_bound * bound_loss)
        return total_loss
