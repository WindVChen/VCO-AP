from torch import nn as nn


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1,
            norm_layer=nn.BatchNorm2d, activation=nn.ELU,
            bias=True,
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )

    def forward(self, x):
        return self.block(x)


class MaxPoolDownSize(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, depth):
        super(MaxPoolDownSize, self).__init__()
        self.depth = depth
        self.reduce_conv = ConvBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.convs = nn.ModuleList([
            ConvBlock(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
            for conv_i in range(depth)
        ])
        self.pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        outputs = []

        output = self.reduce_conv(x)

        for conv_i, conv in enumerate(self.convs):
            output = output if conv_i == 0 else self.pool2d(output)
            outputs.append(conv(output))

        return outputs
