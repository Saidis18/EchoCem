import torch
from typing import List, Tuple
import time
import torch.utils.data


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
        self.block = Block(out_channels * 2, out_channels)
    
    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.up_conv(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.block(x)
        return x


class UNetBase(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: List[int]):
        super(UNetBase, self).__init__() # type: ignore
        dims_down, dims_up = UNetBase._get_dims(features)

        self.first_down = DownBlock(in_channels, features[0])
        self.down_blocks = torch.nn.ModuleList([DownBlock(in_ch, out_ch) for in_ch, out_ch in dims_down])
        self.bottleneck = Block(features[-1], features[-1] * 2)
        self.up_blocks = torch.nn.ModuleList([UpBlock(in_ch, out_ch) for in_ch, out_ch in dims_up])
        self.last_up = UpBlock(features[0] * 2, features[0])
        self.final_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip, x = self.first_down(x)
        skip_connections: List[torch.Tensor] = [skip]
        for down in self.down_blocks:
            skip, x = down(x)
            skip_connections.append(skip)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx, up in enumerate(self.up_blocks):
            x = up(x, skip_connections[idx])
        
        x = self.last_up(x, skip_connections[-1])
        x = self.final_conv(x)
        return x
    
    _dims_t = List[Tuple[int, int]]
    @staticmethod
    def _get_dims(features: List[int]) -> Tuple[_dims_t, _dims_t]:
        dims_down: UNetBase._dims_t = []
        for i in range(len(features) - 1):
            dims_down.append((features[i], features[i + 1]))
        dims_up: UNetBase._dims_t = [(2*ft2, 2*ft1) for ft1, ft2 in reversed(dims_down)]
        return dims_down, dims_up


class UNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: List[int] = [64, 128, 256, 512]):
        super(UNet, self).__init__() # type: ignore
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for UNet")
            self.unet = torch.nn.DataParallel(UNetBase(in_channels, out_channels, features))
        else:
            self.unet = UNetBase(in_channels, out_channels, features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)
    
    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    _dataloader_t = torch.utils.data.DataLoader[torch.Tensor]

    def epoch(self, dataloader: _dataloader_t, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device) -> float:
        self.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(self, dataloader: _dataloader_t, loss_fn: torch.nn.Module, device: torch.device) -> float:
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def training_loop(self, train_dataloader: _dataloader_t, val_dataloader: _dataloader_t, epochs: int, device: torch.device) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            init_time = time.time()
            train_loss = self.epoch(train_dataloader, optimizer, loss_fn, device)
            val_loss = self.validate(val_dataloader, loss_fn, device)
            end_time = time.time()
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {end_time - init_time:.2f}s")


if __name__ == "__main__":
    import data
    from torch.utils.data import random_split, DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=3).to(device)
    print(f"Trainable parameters: {model.param_count}")
    print(f"Using device: {device}")

    DATA_DIR = data.Path(__file__).parent / "data"
    X_DIR = DATA_DIR / "X_train_uDRk9z9" / "images"
    Y_CSV = DATA_DIR / 'Y_train_T9NrBYo.csv'
    dataset = data.EchoCementDataset(X_DIR, Y_CSV)
    dataset = torch.utils.data.Subset(dataset, list(range(64))) 
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    model.training_loop(train_loader, val_loader, epochs=5, device=device)
