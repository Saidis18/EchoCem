import torch
from typing import List, Tuple
import time


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


class UNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: List[int] = [64, 128, 256, 512]):
        super(UNet, self).__init__() # type: ignore
        dims_down, dims_up = UNet._get_dims(features)

        self.first_down = DownBlock(in_channels, features[0])
        self.down_blocks = torch.nn.ModuleList([DownBlock(in_ch, out_ch) for in_ch, out_ch in dims_down])
        self.bottleneck = Block(features[-1], features[-1] * 2)
        self.up_blocks = torch.nn.ModuleList([UpBlock(in_ch, out_ch) for in_ch, out_ch in dims_up])
        self.last_up = UpBlock(features[0] * 2, features[0])
        self.final_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.act = torch.nn.Softmax(dim=1)
    
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
        x = self.act(x)
        return x
    
    _dims_t = List[Tuple[int, int]]
    @staticmethod
    def _get_dims(features: List[int]) -> Tuple[_dims_t, _dims_t]:
        dims_down: UNet._dims_t = []
        for i in range(len(features) - 1):
            dims_down.append((features[i], features[i + 1]))
        dims_up: UNet._dims_t = [(2*ft2, 2*ft1) for ft1, ft2 in reversed(dims_down)]
        return dims_down, dims_up
    
    def epoch(
            self,
            dataloader: torch.utils.data.DataLoader[torch.Tensor],
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            device: torch.device
        ) -> float:
        self.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.long()  # Convert to Long type for CrossEntropyLoss
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def training_loop(self, dataloader: torch.utils.data.DataLoader[torch.Tensor], epochs: int, device: torch.device) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            init_time = time.time()
            epoch_loss = self.epoch(dataloader, optimizer, loss_fn, device)
            end_time = time.time()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Time: {end_time - init_time:.2f}s")


if __name__ == "__main__":
    import data
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=3).to(device)

    DATA_DIR = data.Path(__file__).parent / "data"
    X_DIR = DATA_DIR / "X_train_uDRk9z9" / "images"
    Y_CSV = DATA_DIR / 'Y_train_T9NrBYo.csv'
    dataset = data.EchoCementDataset(X_DIR, Y_CSV)
    sub = torch.utils.data.Subset(dataset, list(range(64)))  # type: ignore
    dataloader = torch.utils.data.DataLoader(sub, batch_size=8, shuffle=True, num_workers=4)

    from torch.profiler import profile, ProfilerActivity

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        record_shapes=True,
        profile_memory=True
    ) as prof:

        model.training_loop(
            dataloader,
            epochs=5,
            device=device
        )
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=200)) # type: ignore


    # model.training_loop(dataloader, epochs=30, device=device)
