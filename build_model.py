import torch
from model.backbone import baseline
from model.lut_transformation_net import build_lut_transform


# LUT Pytorch prediction model structure.
class model_LUT(torch.nn.Module):
    def __init__(self, args):
        super(model_LUT, self).__init__()
        self.fine_feature = baseline(args)

        self.lut_transform = build_lut_transform(64, args.LUT_dim, args)

    def forward(self, local_image, mask):
        fine_feature = self.fine_feature(local_image, mask)

        # Fuse coarse and fine features.
        final_feature = fine_feature

        """3D LUT prediction."""
        fit_lut3d, lut_transform_image = self.lut_transform(local_image, final_feature)

        return fit_lut3d, lut_transform_image
