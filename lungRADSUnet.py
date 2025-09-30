class MultiModal3DNet(nn.Module):
    def __init__(self, n_classes: int, tab_dim: int = 3):
        super().__init__()
        # small 3D CNN trunk (reduce memory footprint)
        self.trunk = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # /2

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # /4

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)  # -> (B,32,1,1,1)
        )
        self.tabular = nn.Sequential(
            nn.Linear(tab_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, vol, tab):
        # vol: (B,1,D,H,W)
        v = self.trunk(vol).view(vol.size(0), -1)  # (B,32)
        t = self.tabular(tab)  # (B,16)
        x = torch.cat([v, t], dim=1)
        return self.classifier(x)