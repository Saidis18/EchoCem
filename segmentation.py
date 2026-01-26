import torch
from typing import List, Tuple
import time
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import config
import torchvision # type: ignore


class Block(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Block, self).__init__() # type: ignore
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class AttentionGate(torch.nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__() # type: ignore
        self.W_g = torch.nn.Sequential(
            torch.nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )
        self.W_x = torch.nn.Sequential(
            torch.nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )
        self.psi = torch.nn.Sequential(
            torch.nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid()
        )
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DownBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownBlock, self).__init__() # type: ignore
        self.block = Block(in_channels, out_channels)
        self.down = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        x = self.down(skip)
        return skip, x


class UpBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpBlock, self).__init__() # type: ignore
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.attention = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        self.block = Block(out_channels * 2, out_channels)
    
    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.up_conv(x)
        skip_connection = self.attention(g=x, x=skip_connection)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.block(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, features: List[int], dims_down: List[Tuple[int, int]]):
        super(Encoder, self).__init__() # type: ignore

        self.first_down = DownBlock(in_channels, features[0])
        self.down_blocks = torch.nn.ModuleList([DownBlock(in_ch, out_ch) for in_ch, out_ch in dims_down])
        self.bottleneck = Block(features[-1], features[-1] * 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip, x = self.first_down(x)
        skip_connections: List[torch.Tensor] = [skip]
        for down in self.down_blocks:
            skip, x = down(x)
            skip_connections.append(skip)
        x = self.bottleneck(x)
        return x, skip_connections[::-1]


class Decoder(torch.nn.Module):
    def __init__(self, features: List[int], out_channels: int, dims_up: List[Tuple[int, int]]):
        super(Decoder, self).__init__() # type: ignore

        self.up_blocks = torch.nn.ModuleList([UpBlock(in_ch, out_ch) for in_ch, out_ch in dims_up])
        self.supervision = torch.nn.ModuleList([torch.nn.Conv2d(out_ch, out_channels, kernel_size=1) for _, out_ch in dims_up])
        self.last_up = UpBlock(features[0] * 2, features[0])
    
    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        supervised_latents: List[torch.Tensor] = []
        for idx, up in enumerate(self.up_blocks):
            x = up(x, skip_connections[idx])
            supervised_latents.append(self.supervision[idx](x))
        x = self.last_up(x, skip_connections[-1])
        return x, supervised_latents

class UNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: List[int]):
        super(UNet, self).__init__() # type: ignore
        dims_down, dims_up = UNet._get_dims(features)

        self.encoder = Encoder(in_channels, features, dims_down)
        self.decoder = Decoder(features, out_channels, dims_up)
        self.final_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x, skip_connections = self.encoder(x)
        x, supervised_latents = self.decoder(x, skip_connections)
        x = self.final_conv(x)
        return x, supervised_latents[::-1]
    
    _dims_t = List[Tuple[int, int]]
    @staticmethod
    def _get_dims(features: List[int]) -> Tuple[_dims_t, _dims_t]:
        dims_down: UNet._dims_t = []
        for i in range(len(features) - 1):
            dims_down.append((features[i], features[i + 1]))
        dims_up: UNet._dims_t = [(2*ft2, 2*ft1) for ft1, ft2 in reversed(dims_down)]
        return dims_down, dims_up


class Segmentation(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, conf: config.Config):
        super(Segmentation, self).__init__() # type: ignore
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for segmentation")
            self.base_model = torch.nn.DataParallel(base_model)
        else:
            self.base_model = base_model
        self.loss_fn = conf.loss_fn
        self.conf = conf
        self.epoch_count = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
    
    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    _dataloader_t = torch.utils.data.DataLoader[torch.Tensor]

    def epoch(self, dataloader: _dataloader_t, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device) -> float:
        self.epoch_count += 1
        self.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, supervised_latents = self(inputs)
            loss = loss_fn(outputs, targets)
            loss += self.downsample_loss(supervised_latents, targets, loss_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
        return total_loss / len(dataloader)
    
    def downsample_loss(self, supervised_latents: List[torch.Tensor], targets: torch.Tensor, loss_fn: torch.nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0).to(targets.device)
        for i, sup_out in enumerate(supervised_latents):
            if targets.shape[1] == 1:
                targets_for_interp = targets.float()
            else:
                targets_for_interp = targets.unsqueeze(1).float()
            downsampled_target = torch.nn.functional.interpolate(targets_for_interp, size=sup_out.shape[2:], mode='nearest').squeeze(1).round().to(targets.dtype)
            if sup_out.shape[1] == 1:
                sup_out = sup_out.squeeze(1)
            loss += loss_fn(sup_out, downsampled_target) / (4 ** i)
        return loss

    def validate(self, dataloader: _dataloader_t, loss_fn: torch.nn.Module, device: torch.device) -> float:
        if self.conf.TEST_RATIO == 0.0:
            return 0.0
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = self(inputs)
                # Ensure targets have the correct shape for loss computation
                if targets.dim() == 2:
                    targets_for_loss = targets.unsqueeze(1)
                else:
                    targets_for_loss = targets
                loss = loss_fn(outputs, targets_for_loss)
                total_loss += loss.detach()
                self.log_image(inputs, outputs, targets)
        return total_loss / len(dataloader)
    
    def log_image(self, inputs: torch.Tensor, model_output: torch.Tensor, targets: torch.Tensor) -> None:
        idx = np.random.randint(0, model_output.shape[0], (1,)).item()
        try:
            if model_output.shape[1] > 1:
                prediction = model_output.argmax(dim=1).squeeze().detach().cpu()[idx]
                out_name = f"log/train_{self.epoch_count}.png"
            else:
                prediction = model_output.squeeze().detach().cpu()[idx]
                out_name = f"log/pre_train_{self.epoch_count}.png"
            _, axarr = plt.subplots(3, 1) # type: ignore
            axarr[0].imshow(prediction)
            axarr[1].imshow(targets.squeeze().detach().cpu()[idx])
            axarr[2].imshow(inputs.squeeze().detach().cpu()[idx])
            plt.savefig(out_name, dpi=300, bbox_inches="tight") # type: ignore
            plt.close()
        except:
            pass
        finally:
            plt.close('all')
    
    def training_loop(self, train_dataloader: _dataloader_t, val_dataloader: _dataloader_t, epochs: int, device: torch.device) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = self.loss_fn
        for _ in range(epochs):
            init_time = time.time()
            train_loss = self.epoch(train_dataloader, optimizer, loss_fn, device)
            val_loss = self.validate(val_dataloader, loss_fn, device)
            scheduler.step()
            end_time = time.time()
            print(f"Epoch {self.epoch_count}/{self.conf.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {end_time - init_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")

    def predict(self, img: np.ndarray, device: torch.device) -> torch.Tensor:
        self.eval()
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # type: ignore
        x = (x - x.mean()) / (x.std() + 1e-8)
        x = torch.nn.functional.interpolate(x, size=(160, 160), mode='nearest')
        with torch.no_grad():
            logits, _ = self(x)
            logits = torch.nn.functional.interpolate(logits, size=img.shape, mode='bilinear')
            predictions = logits.argmax(dim=1)
        return predictions.cpu() # type: ignore
