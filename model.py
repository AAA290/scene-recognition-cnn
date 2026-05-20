import torch
import torch.nn as nn


class SceneCNN(nn.Module):
    """
    Lightweight VGG-style CNN for 15-class scene classification.

    The backbone consists of four double-convolution blocks with progressive
    channel expansion (32 -> 64 -> 128 -> 256). Global Average Pooling
    replaces a flat fully-connected head to reduce parameter count and
    suppress overfitting on the small 1,500-image training set.
    """

    def __init__(self, num_classes=15):
        super().__init__()

        def _make_block(in_c, out_c):
            """
            Build a standard [Conv-BN-ReLU] x2 + MaxPool block.

            Two consecutive 3x3 convolutions provide an effective receptive
            field equivalent to a single 5x5 convolution while introducing
            more non-linearity with fewer parameters.
            """
            return nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # Feature extractor: channel width doubles at each block.
        self.features = nn.Sequential(
            _make_block(in_c=3,   out_c=32),   # [B,   3, 224, 224] -> [B,  32, 112, 112]
            _make_block(in_c=32,  out_c=64),   # [B,  32, 112, 112] -> [B,  64,  56,  56]
            _make_block(in_c=64,  out_c=128),  # [B,  64,  56,  56] -> [B, 128,  28,  28]
            _make_block(in_c=128, out_c=256),  # [B, 128,  28,  28] -> [B, 256,  14,  14]
        )

        # Global Average Pooling collapses spatial dimensions to 1x1,
        # acting as a structural regulariser that drastically cuts parameters.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # MLP classification head with two Dropout layers to further prevent overfitting.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # [B, 256, 1, 1] -> [B, 256]
        out = self.classifier(x)
        return out

    def _initialize_weights(self):
        """
        Apply principled weight initialisation to accelerate convergence.

        Kaiming Normal initialisation is used for Conv2d layers since it is
        derived under the assumption of ReLU activations, compensating for
        the variance reduction caused by zeroing negative activations.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = SceneCNN(num_classes=15)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Expected: [2, 15]