import torch
import math


class RandomZeroedPatch:
        def __init__(self, patch_size: int = 50):
            self.patch_size = patch_size
        
        def __call__(self, img: torch.Tensor) -> torch.Tensor:
            h, w = img.shape[-2:]
            x = torch.randint(0, h - self.patch_size, (1,)).item()
            y = torch.randint(0, w - self.patch_size, (1,)).item()
            img = img.clone()
            img[..., x:x+self.patch_size, y:y+self.patch_size] = 0
            return img


class Augmentation:
    def __init__(self, rot_angle: float=5.0):
        self.flip_hori = torch.rand(size=(1,), dtype=torch.float32).item() < 0.5
        self.flip_vert = torch.rand(size=(1,), dtype=torch.float32).item() < 0.5
        self.rotation_angle = (torch.rand(size=(1,), dtype=torch.float32).item() * 2 * rot_angle) - rot_angle

    def _rotate_tensor(self, img: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate a tensor using affine transformation."""
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Affine matrix for rotation around center
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        # Add batch dimension if needed
        if img.dim() == 3:
            img = img.unsqueeze(0)
            remove_batch = True
        else:
            remove_batch = False
        
        grid = torch.nn.functional.affine_grid(theta, img.shape) # type: ignore
        rotated = torch.nn.functional.grid_sample(img, grid, mode='nearest', padding_mode='zeros')
        
        if remove_batch:
            rotated = rotated.squeeze(0)
        
        return rotated

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(160, 160), mode='nearest').squeeze(0)
        if self.flip_hori:
            img = torch.flip(img, dims=[2])
        if self.flip_vert:
            img = torch.flip(img, dims=[1])
        img = self._rotate_tensor(img, self.rotation_angle)
        return img # type: ignore
