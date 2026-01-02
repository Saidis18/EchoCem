import torch


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
