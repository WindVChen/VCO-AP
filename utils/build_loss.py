import torch
import numpy as np


def loss_generator(ignore: list = None):
    loss_fn = {'mse': mse,
               'lut_mse': lut_mse,
               'masked_mse': masked_mse,
               'sample_weighted_mse': sample_weighted_mse,
               'regularize_LUT': regularize_LUT,
               'MaskWeightedMSE': MaskWeightedMSE,
               'TV_3D': TV_3D,
               'TVLoss': TVLoss,
               'NPSCalculator': NPSCalculator}

    if ignore:
        for fn in ignore:
            ignore.pop(fn)

    return loss_fn


def mse(pred, gt, mask=None):
    if mask is not None:
        mask = mask.view(mask.shape[0], 1, 1, 1, 1)
        mask = ~mask * torch.ones_like(pred)
        return torch.sum((pred - gt) ** 2 * mask) / torch.sum(mask)
    return torch.mean((pred - gt) ** 2)


def masked_mse(pred, gt, mask):
    delimin = torch.clamp_min(torch.sum(mask, dim=([x for x in range(1, len(mask.shape))])), 100).cuda()
    # total = torch.sum(torch.ones_like(mask), dim=([x for x in range(1, len(mask.shape))]))
    out = torch.sum((mask > 100 / 255.) * (pred - gt) ** 2, dim=([x for x in range(1, len(mask.shape))]))
    out = out / delimin
    return torch.mean(out)


def sample_weighted_mse(pred, gt, mask):
    multi_factor = torch.clamp_min(torch.sum(mask, dim=([x for x in range(1, len(mask.shape))])), 100).cuda()
    multi_factor = multi_factor / (multi_factor.sum())
    # total = torch.sum(torch.ones_like(mask), dim=([x for x in range(1, len(mask.shape))]))
    out = torch.mean((pred - gt) ** 2, dim=([x for x in range(1, len(mask.shape))]))
    out = out * multi_factor
    return torch.sum(out)


def regularize_LUT(lut):
    st = lut[lut < 0.]
    reg_st = (st ** 2).mean() if min(st.shape) != 0 else 0

    lt = lut[lut > 1.]
    reg_lt = ((lt - 1.) ** 2).mean() if min(lt.shape) != 0 else 0

    return reg_lt + reg_st


def lut_mse(feat, lut_batch):
    loss = 0
    for id in range(feat.shape[0] // lut_batch):
        for i in feat[id * lut_batch: id * lut_batch + lut_batch]:
            for j in feat[id * lut_batch: id * lut_batch + lut_batch]:
                loss += mse(i, j)

    return loss / lut_batch


def MaskWeightedMSE(pred, label, mask):
    label = label.view(pred.size())
    reduce_dims = get_dims_with_exclusion(label.dim(), 0)

    loss = (pred - label) ** 2
    delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask, dim=reduce_dims), 100)
    loss = torch.sum(loss, dim=reduce_dims) / delimeter

    return torch.mean(loss)


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


class TV_3D(torch.nn.Module):
    def __init__(self, dim=33, device="cuda"):
        super(TV_3D, self).__init__()

        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float).to(device)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float).to(device)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float).to(device)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU().to(device)

    def forward(self, LUT):
        dif_r = LUT[:, :, :, :, :-1] - LUT[:, :, :, :, 1:]
        dif_g = LUT[:, :, :, :-1, :] - LUT[:, :, :, 1:, :]
        dif_b = LUT[:, :, :-1, :, :] - LUT[:, :, 1:, :, :]
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn


class TVLoss(torch.nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class NPSCalculator(torch.nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = torch.nn.Parameter(self.get_printability_array(printability_file, patch_side),
                                                     requires_grad=False).cuda()

    def forward(self, adv_patchs):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        loss = 0.
        for adv_patch in adv_patchs:
            color_dist = (adv_patch.unsqueeze(0) - self.printability_array + 0.000001)
            color_dist = color_dist ** 2
            color_dist = torch.sum(color_dist, 1) + 0.000001
            color_dist = torch.sqrt(color_dist)
            # only work with the min distance
            color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
            # calculate the nps by summing over all pixels
            nps_score = torch.sum(color_dist_prod, 0)
            nps_score = torch.sum(nps_score, 0)
            loss += nps_score / torch.numel(adv_patch)
        return loss

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa
