import torch
import torch.nn as nn

from .conv_autoencoder import ConvEncoder, LUTDecoder

from .ops import ScaleLayer


class IHModelWithBackbone(nn.Module):
    def __init__(
            self,
            model, backbone,
            downsize_backbone_input=False,
            mask_fusion='sum',
            backbone_conv1_channels=64, args=None
    ):
        super(IHModelWithBackbone, self).__init__()
        self.downsize_backbone_input = downsize_backbone_input
        self.mask_fusion = mask_fusion

        self.backbone = backbone
        self.model = model
        self.args = args

        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, backbone_conv1_channels, kernel_size=3, stride=2, padding=1, bias=True),
            ScaleLayer(init_value=0.1, lr_mult=1)
        )

    def forward(self, image, mask):
        backbone_image = image

        backbone_mask = torch.cat((mask, 1.0 - mask), dim=1)
        backbone_mask_features = self.mask_conv(backbone_mask[:, :1])

        backbone_features = self.backbone(backbone_image, backbone_mask, backbone_mask_features)

        output = self.model(image, mask, backbone_features)
        return output


class DeepImageHarmonization(nn.Module):
    def __init__(
            self,
            depth,
            norm_layer=nn.BatchNorm2d, batchnorm_from=0,
            ch=64, max_channels=512,
            backbone_from=-1, backbone_channels=None, backbone_mode='', args=None
    ):
        super(DeepImageHarmonization, self).__init__()
        self.depth = depth
        self.encoder = ConvEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.args = args
        self.decoder = LUTDecoder(depth, self.encoder.blocks_channels, norm_layer, args, backbone_from)

    def forward(self, image, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)

        intermediates = self.encoder(x, backbone_features)

        lut_feature = self.decoder(intermediates)
        return lut_feature
