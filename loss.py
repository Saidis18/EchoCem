import torch


class WeightedCELoss(torch.nn.Module):
    """Weighted CrossEntropyLoss with ignore_index support for older PyTorch."""

    def __init__(self, weight: torch.Tensor, ignore_index: int=-100):
        super(WeightedCELoss, self).__init__()  # type: ignore
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Create mask for valid pixels (not ignored)
        mask = targets != self.ignore_index

        # Replace ignored indices with 0 temporarily (valid class index)
        targets_safe = targets.clone()
        targets_safe[~mask] = 0

        # Compute per-pixel CE loss without reduction
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets_safe, reduction="none")

        # Apply class weights manually
        # targets_safe is [batch, height, width], weight is [3]
        # We need to index weight with targets_safe to get a [batch, height, width] weight map
        # Use view to handle the indexing safely
        weight_map = self.weight[targets_safe.long()]
        weighted_loss = ce_loss * weight_map

        # Apply mask and compute mean over valid pixels only
        weighted_loss = weighted_loss * mask.float()
        total_valid = mask.float().sum().clamp(min=1)
        return weighted_loss.sum() / total_valid


class DiceCELoss(torch.nn.Module):
    def __init__(self, weight_ce: float = 0.5, smooth: float = 1e-6):
        super(DiceCELoss, self).__init__() # type: ignore
        self.weight_ce = weight_ce
        self.smooth = smooth
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs_soft = torch.nn.functional.softmax(inputs, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(inputs_soft * targets_one_hot)
        cardinality = torch.sum(inputs_soft + targets_one_hot)
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
        ax2 = torch.abs(x[:, :, 1:] - x[:, :, :-1]).to(torch.float32)
        return torch.mean(ax1) + torch.mean(ax2)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        tv_loss = self.tv_loss(inputs)
        total_loss = ce_loss + self.tv_weight * tv_loss
        return total_loss
