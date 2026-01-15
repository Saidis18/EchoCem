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
