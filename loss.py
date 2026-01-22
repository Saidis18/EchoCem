import torch


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
    """GPU-friendly boundary loss.

    This implementation avoids CPU distance transforms (SciPy/NumPy) and instead
    matches predicted vs target boundaries using a morphological gradient and a
    Dice-style overlap loss.
    """
    def __init__(
        self,
        ignore_index: int = -100,
        kernel_size: int = 3,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ):
        super(BoundaryLoss, self).__init__() # type: ignore
        self.ignore_index = ignore_index
        self.kernel_size = kernel_size
        self.smooth = smooth
        self.reduction = reduction

    def _morphological_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Morphological gradient (dilation - erosion) with min/max pooling."""
        k = int(self.kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 1")
        if k == 1:
            return torch.zeros_like(x)

        pad = k // 2
        dil = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
        ero = -torch.nn.functional.max_pool2d(-x, kernel_size=k, stride=1, padding=pad)
        return (dil - ero).clamp_min(0.0)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 4:
            raise ValueError("BoundaryLoss expects inputs of shape (B, C, H, W)")
        if targets.dim() != 3:
            raise ValueError("BoundaryLoss expects targets of shape (B, H, W)")

        num_classes = inputs.shape[1]
        probs = torch.nn.functional.softmax(inputs, dim=1)

        valid_mask = (targets != self.ignore_index) & (targets >= 0) & (targets < num_classes)
        valid_mask_f = valid_mask.unsqueeze(1).to(dtype=probs.dtype)

        # Safe one-hot encoding for targets
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        target_1h = torch.nn.functional.one_hot(targets_safe, num_classes=num_classes).permute(0, 3, 1, 2).to(dtype=probs.dtype)
        target_1h = target_1h * valid_mask_f

        with torch.no_grad():
            target_boundary = self._morphological_gradient(target_1h)

        pred_boundary = self._morphological_gradient(probs) * valid_mask_f

        # Boundary Dice loss (global over batch/classes/pixels)
        intersection = torch.sum(pred_boundary * target_boundary)
        cardinality = torch.sum(pred_boundary) + torch.sum(target_boundary)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice

        if self.reduction == "none":
            return (pred_boundary - target_boundary).abs()
        if self.reduction == "sum":
            # Keep historical meaning: scale by number of valid pixels
            return loss * (valid_mask_f.sum() + 1e-8)
        return loss


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
