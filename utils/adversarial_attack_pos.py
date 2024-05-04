import kornia as K
import torch
from mmdet.apis.test import normalize
import numpy as np

def addPert_center(pert, img, pos, downsize=5, adapative_transform=True):
    """
    Add four attack masks to an object's two sides
    :param pert: adv_patch
    :param img:  target image
    :param pos:  [8,] gt rotated bbox. 4 pairs of coordinates in clock-wise direction.
    :param downsize:  downsize factor (only work when adapative_transform is False)
    :param adapative_transform:  when it is true, the downsize factor is computed based on the adv_patch size
    :return:  perturbed image
    """
    inv_pert = normalize(pert, 'inv')
    inv_img = normalize(img, 'inv')

    h, w = pert.shape[-2:]

    downsize = 2*downsize

    bbox_oneSide = np.linalg.norm(np.array([pos[0] - pos[2], pos[1] - pos[3]]))
    bbox_anotherSide = np.linalg.norm(np.array([pos[0] - pos[6], pos[1] - pos[7]]))
    if adapative_transform:
        downsize_h = 2 * bbox_oneSide / h
        downsize_w = 2 * bbox_anotherSide / w
        tpos1 = torch.tensor([[
            [((pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize_h + ((pos[6] + pos[4]) / 2 - (
                    pos[4] - pos[6]) / downsize_h)) / 2 +
             ((pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize_h - ((pos[6] + pos[4]) / 2 - (
                     pos[4] - pos[6]) / downsize_h)) / downsize_w,
             ((pos[1] + pos[3]) / 2 - (pos[3] - pos[1]) / downsize_h + ((pos[7] + pos[5]) / 2 - (
                     pos[5] - pos[7]) / downsize_h)) / 2 +
             ((pos[1] + pos[3]) / 2 - (pos[3] - pos[1]) / downsize_h - ((pos[7] + pos[5]) / 2 - (
                     pos[5] - pos[7]) / downsize_h)) / downsize_w],
            [((pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize_h + ((pos[6] + pos[4]) / 2 + (
                    pos[4] - pos[6]) / downsize_h)) / 2 +
             ((pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize_h - ((pos[6] + pos[4]) / 2 + (
                     pos[4] - pos[6]) / downsize_h)) / downsize_w,
             ((pos[1] + pos[3]) / 2 + (pos[3] - pos[1]) / downsize_h + ((pos[7] + pos[5]) / 2 + (
                     pos[5] - pos[7]) / downsize_h)) / 2 +
             ((pos[1] + pos[3]) / 2 + (pos[3] - pos[1]) / downsize_h - ((pos[7] + pos[5]) / 2 + (
                     pos[5] - pos[7]) / downsize_h)) / downsize_w],
            [((pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize_h + ((pos[6] + pos[4]) / 2 + (
                    pos[4] - pos[6]) / downsize_h)) / 2 -
             ((pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize_h - ((pos[6] + pos[4]) / 2 + (
                     pos[4] - pos[6]) / downsize_h)) / downsize_w,
             ((pos[1] + pos[3]) / 2 + (pos[3] - pos[1]) / downsize_h + ((pos[7] + pos[5]) / 2 + (
                     pos[5] - pos[7]) / downsize_h)) / 2 -
             ((pos[1] + pos[3]) / 2 + (pos[3] - pos[1]) / downsize_h - ((pos[7] + pos[5]) / 2 + (
                     pos[5] - pos[7]) / downsize_h)) / downsize_w],
            [((pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize_h + ((pos[6] + pos[4]) / 2 - (
                    pos[4] - pos[6]) / downsize_h)) / 2 -
             ((pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize_h - ((pos[6] + pos[4]) / 2 - (
                     pos[4] - pos[6]) / downsize_h)) / downsize_w,
             ((pos[1] + pos[3]) / 2 - (pos[3] - pos[1]) / downsize_h + ((pos[7] + pos[5]) / 2 - (
                     pos[5] - pos[7]) / downsize_h)) / 2 -
             ((pos[1] + pos[3]) / 2 - (pos[3] - pos[1]) / downsize_h - ((pos[7] + pos[5]) / 2 - (
                     pos[5] - pos[7]) / downsize_h)) / downsize_w],
        ]], dtype=torch.float32)
    else:
        tpos1 = torch.tensor([[
            [((pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize + ((pos[6] + pos[4]) / 2 - (
                        pos[4] - pos[6]) / downsize)) / 2 +
             ((pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize - ((pos[6] + pos[4]) / 2 - (
                         pos[4] - pos[6]) / downsize)) / downsize,
             ((pos[1] + pos[3]) / 2 - (pos[3] - pos[1]) / downsize + ((pos[7] + pos[5]) / 2 - (
                         pos[5] - pos[7]) / downsize)) / 2 +
             ((pos[1] + pos[3]) / 2 - (pos[3] - pos[1]) / downsize - ((pos[7] + pos[5]) / 2 - (
                         pos[5] - pos[7]) / downsize)) / downsize],
            [((pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize + ((pos[6] + pos[4]) / 2 + (
                        pos[4] - pos[6]) / downsize)) / 2 +
             ((pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize - ((pos[6] + pos[4]) / 2 + (
                         pos[4] - pos[6]) / downsize)) / downsize,
             ((pos[1] + pos[3]) / 2 + (pos[3] - pos[1]) / downsize + ((pos[7] + pos[5]) / 2 + (
                         pos[5] - pos[7]) / downsize)) / 2 +
             ((pos[1] + pos[3]) / 2 + (pos[3] - pos[1]) / downsize - ((pos[7] + pos[5]) / 2 + (
                         pos[5] - pos[7]) / downsize)) / downsize],
            [((pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize + ((pos[6] + pos[4]) / 2 + (
                        pos[4] - pos[6]) / downsize)) / 2 -
             ((pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize - ((pos[6] + pos[4]) / 2 + (
                         pos[4] - pos[6]) / downsize)) / downsize,
             ((pos[1] + pos[3]) / 2 + (pos[3] - pos[1]) / downsize + ((pos[7] + pos[5]) / 2 + (
                         pos[5] - pos[7]) / downsize)) / 2 -
             ((pos[1] + pos[3]) / 2 + (pos[3] - pos[1]) / downsize - ((pos[7] + pos[5]) / 2 + (
                         pos[5] - pos[7]) / downsize)) / downsize],
            [((pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize + ((pos[6] + pos[4]) / 2 - (
                    pos[4] - pos[6]) / downsize)) / 2 -
             ((pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize - ((pos[6] + pos[4]) / 2 - (
                     pos[4] - pos[6]) / downsize)) / downsize,
             ((pos[1] + pos[3]) / 2 - (pos[3] - pos[1]) / downsize + ((pos[7] + pos[5]) / 2 - (
                     pos[5] - pos[7]) / downsize)) / 2 -
             ((pos[1] + pos[3]) / 2 - (pos[3] - pos[1]) / downsize - ((pos[7] + pos[5]) / 2 - (
                     pos[5] - pos[7]) / downsize)) / downsize],
        ]], dtype=torch.float32)

    spos = torch.tensor([[
        [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
    ]], dtype=torch.float32)
    M1 = K.geometry.get_perspective_transform(spos, tpos1).cuda()

    mask1 = K.geometry.warp_perspective(torch.ones_like(pert[0]).unsqueeze(0).float().cuda(), M1,
                                        dsize=(img.shape[-2], img.shape[-1]), mode='nearest')

    pert1 = K.geometry.warp_perspective(inv_pert[0].unsqueeze(0).float(), M1, dsize=(img.shape[-2], img.shape[-1]))

    inv_perturbed_img = inv_img * (mask1 != 1) + \
                    pert1 * (mask1 == 1)

    perturbed_img = normalize(inv_perturbed_img)

    mask_all = mask1

    return perturbed_img, mask_all[:, 0:1]

def addPert_corner(pert, img, pos, downsize=3, adapative_transform=True):
    """
    Add four attack masks to an object's four corner
    :param pert: adv_patch
    :param img:  target image
    :param pos:  [8,] gt rotated bbox. 4 pairs of coordinates in clock-wise direction.
    :param downsize:  downsize factor (only work when adapative_transform is False)
    :param adapative_transform:  when it is true, the downsize factor is computed based on the adv_patch size
    :return:  perturbed image
    """
    inv_pert = normalize(pert, 'inv')
    inv_img = normalize(img, 'inv')

    h, w = pert.shape[-2:]
    if adapative_transform:
        bbox_oneSide = np.linalg.norm(np.array([pos[0] - pos[2], pos[1] - pos[3]]))
        bbox_anotherSide = np.linalg.norm(np.array([pos[0] - pos[6], pos[1] - pos[7]]))
        downsize_h = bbox_oneSide/h
        downsize_w = bbox_anotherSide/w
        tpos1 = torch.tensor([[
            [pos[0], pos[1]], [(pos[2] - pos[0]) / downsize_h + pos[0], (pos[3] - pos[1]) / downsize_h + pos[1]],
            [(pos[2] - pos[0]) / downsize_h + (pos[6] - pos[0]) / downsize_w + pos[0],
             (pos[3] - pos[1]) / downsize_h + (pos[7] - pos[1]) / downsize_w + pos[1]],
            [(pos[6] - pos[0]) / downsize_w + pos[0], (pos[7] - pos[1]) / downsize_w + pos[1]],
        ]], dtype=torch.float32)
        tpos2 = torch.tensor([[
            [-(pos[2] - pos[0]) / downsize_h + pos[2], -(pos[3] - pos[1]) / downsize_h + pos[3]], [pos[2], pos[3]],
            [(pos[4] - pos[2]) / downsize_w + pos[2], (pos[5] - pos[3]) / downsize_w + pos[3]],
            [(pos[4] - pos[2]) / downsize_w - (pos[2] - pos[0]) / downsize_h + pos[2],
             (pos[5] - pos[3]) / downsize_w - (pos[3] - pos[1]) / downsize_h + pos[3]],
        ]], dtype=torch.float32)
        tpos3 = torch.tensor([[
            [(pos[6] - pos[4]) / downsize_h + (pos[2] - pos[4]) / downsize_w + pos[4],
             (pos[7] - pos[5]) / downsize_h + (pos[3] - pos[5]) / downsize_w + pos[5]],
            [(pos[2] - pos[4]) / downsize_w + pos[4], (pos[3] - pos[5]) / downsize_w + pos[5]],
            [pos[4], pos[5]],
            [(pos[6] - pos[4]) / downsize_h + pos[4], (pos[7] - pos[5]) / downsize_h + pos[5]],
        ]], dtype=torch.float32)
        tpos4 = torch.tensor([[
            [(pos[0] - pos[6]) / downsize_w + pos[6], (pos[1] - pos[7]) / downsize_w + pos[7]],
            [(pos[4] - pos[6]) / downsize_h + (pos[0] - pos[6]) / downsize_w + pos[6],
             (pos[5] - pos[7]) / downsize_h + (pos[1] - pos[7]) / downsize_w + pos[7]],
            [(pos[4] - pos[6]) / downsize_h + pos[6], (pos[5] - pos[7]) / downsize_h + pos[7]],
            [pos[6], pos[7]],
        ]], dtype=torch.float32)
    else:
        tpos1 = torch.tensor([[
            [pos[0], pos[1]], [(pos[2] - pos[0]) / downsize + pos[0], (pos[3] - pos[1]) / downsize + pos[1]],
            [(pos[2] - pos[0]) / downsize + (pos[6] - pos[0]) / downsize + pos[0],
             (pos[3] - pos[1]) / downsize + (pos[7] - pos[1]) / downsize + pos[1]],
            [(pos[6] - pos[0]) / downsize + pos[0], (pos[7] - pos[1]) / downsize + pos[1]],
        ]], dtype=torch.float32)
        tpos2 = torch.tensor([[
            [-(pos[2] - pos[0]) / downsize + pos[2], -(pos[3] - pos[1]) / downsize + pos[3]], [pos[2], pos[3]],
            [(pos[4] - pos[2]) / downsize + pos[2], (pos[5] - pos[3]) / downsize + pos[3]],
            [(pos[4] - pos[2]) / downsize - (pos[2] - pos[0]) / downsize + pos[2],
             (pos[5] - pos[3]) / downsize - (pos[3] - pos[1]) / downsize + pos[3]],
        ]], dtype=torch.float32)
        tpos3 = torch.tensor([[
            [(pos[6] - pos[4]) / downsize + (pos[2] - pos[4]) / downsize + pos[4],
             (pos[7] - pos[5]) / downsize + (pos[3] - pos[5]) / downsize + pos[5]],
            [(pos[2] - pos[4]) / downsize + pos[4], (pos[3] - pos[5]) / downsize + pos[5]],
            [pos[4], pos[5]],
            [(pos[6] - pos[4]) / downsize + pos[4], (pos[7] - pos[5]) / downsize + pos[5]],
        ]], dtype=torch.float32)
        tpos4 = torch.tensor([[
            [(pos[0] - pos[6]) / downsize + pos[6], (pos[1] - pos[7]) / downsize + pos[7]],
            [(pos[4] - pos[6]) / downsize + (pos[0] - pos[6]) / downsize + pos[6],
             (pos[5] - pos[7]) / downsize + (pos[1] - pos[7]) / downsize + pos[7]],
            [(pos[4] - pos[6]) / downsize + pos[6], (pos[5] - pos[7]) / downsize + pos[7]],
            [pos[6], pos[7]],
        ]], dtype=torch.float32)
    spos = torch.tensor([[
        [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
    ]], dtype=torch.float32)
    M1 = K.geometry.get_perspective_transform(spos, tpos1).cuda()
    M2 = K.geometry.get_perspective_transform(spos, tpos2).cuda()
    M3 = K.geometry.get_perspective_transform(spos, tpos3).cuda()
    M4 = K.geometry.get_perspective_transform(spos, tpos4).cuda()

    mask1 = K.geometry.warp_perspective(torch.ones_like(pert[0]).unsqueeze(0).float().cuda(), M1,
                                        dsize=(img.shape[-2], img.shape[-1]), mode='nearest')
    mask2 = K.geometry.warp_perspective(torch.ones_like(pert[1]).unsqueeze(0).float().cuda(), M2,
                                        dsize=(img.shape[-2], img.shape[-1]), mode='nearest')
    mask3 = K.geometry.warp_perspective(torch.ones_like(pert[2]).unsqueeze(0).float().cuda(), M3,
                                        dsize=(img.shape[-2], img.shape[-1]), mode='nearest')
    mask4 = K.geometry.warp_perspective(torch.ones_like(pert[3]).unsqueeze(0).float().cuda(), M4,
                                        dsize=(img.shape[-2], img.shape[-1]), mode='nearest')

    pert1 = K.geometry.warp_perspective(inv_pert[0].unsqueeze(0).float(), M1, dsize=(img.shape[-2], img.shape[-1]))
    pert2 = K.geometry.warp_perspective(inv_pert[1].unsqueeze(0).float(), M2, dsize=(img.shape[-2], img.shape[-1]))
    pert3 = K.geometry.warp_perspective(inv_pert[2].unsqueeze(0).float(), M3, dsize=(img.shape[-2], img.shape[-1]))
    pert4 = K.geometry.warp_perspective(inv_pert[3].unsqueeze(0).float(), M4, dsize=(img.shape[-2], img.shape[-1]))

    inv_perturbed_img = inv_img * (mask1 != 1) * (mask2 != 1) * (mask3 != 1) * (mask4 != 1) + \
                    pert1 * (mask1 == 1) * (mask2 != 1) * (mask3 != 1) * (mask4 != 1) + \
                        pert2 * (mask2 == 1) * (mask1 != 1) * (mask3 != 1) * (mask4 != 1) +\
                        pert3 * (mask3 == 1) * (mask1 != 1) * (mask2 != 1) * (mask4 != 1) +\
                        pert4 * (mask4 == 1) * (mask1 != 1) * (mask2 != 1) * (mask3 != 1)

    perturbed_img = normalize(inv_perturbed_img)

    mask_all = mask1 + mask2 + mask3 + mask4

    return perturbed_img, mask_all[:, 0:1]


def addPert_lateral(pert, img, pos, downsize=10, adapative_transform=True):
    """
    Add four attack masks to an object's two sides
    :param pert: adv_patch
    :param img:  target image
    :param pos:  [8,] gt rotated bbox. 4 pairs of coordinates in clock-wise direction.
    :param downsize:  downsize factor (only work when adapative_transform is False)
    :param adapative_transform:  when it is true, the downsize factor is computed based on the adv_patch size
    :return:  perturbed image
    """
    inv_pert = normalize(pert, 'inv')
    inv_img = normalize(img, 'inv')

    h, w = pert.shape[-2:]

    bbox_oneSide = np.linalg.norm(np.array([pos[0] - pos[2], pos[1] - pos[3]]))
    bbox_anotherSide = np.linalg.norm(np.array([pos[0] - pos[6], pos[1] - pos[7]]))
    if adapative_transform:
        downsize_h = bbox_oneSide / h
        downsize_w = bbox_anotherSide / w
        if bbox_oneSide > bbox_anotherSide:
            downsize_h = 2 * downsize_h
            tpos1 = torch.tensor([[
                [(pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize_h,
                 (pos[1] + pos[3]) / 2 - (pos[3] - pos[1]) / downsize_h],
                [(pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize_h,
                 (pos[1] + pos[3]) / 2 + (pos[3] - pos[1]) / downsize_h],
                [((pos[6] + pos[4]) / 2 + (pos[4] - pos[6]) / downsize_h - ((pos[0] + pos[2]) / 2 + (
                        pos[2] - pos[0]) / downsize_h)) / downsize_w + (pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize_h,
                 ((pos[7] + pos[5]) / 2 + (pos[5] - pos[7]) / downsize_h - ((pos[1] + pos[3]) / 2 + (
                         pos[3] - pos[1]) / downsize_h)) / downsize_w + (pos[1] + pos[3]) / 2 + (
                         pos[3] - pos[1]) / downsize_h],
                [((pos[6] + pos[4]) / 2 - (pos[4] - pos[6]) / downsize_h - ((pos[0] + pos[2]) / 2 - (
                        pos[2] - pos[0]) / downsize_h)) / downsize_w + (pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize_h,
                 ((pos[7] + pos[5]) / 2 - (pos[5] - pos[7]) / downsize_h - ((pos[1] + pos[3]) / 2 - (
                         pos[3] - pos[1]) / downsize_h)) / downsize_w + (pos[1] + pos[3]) / 2 - (
                         pos[3] - pos[1]) / downsize_h],
            ]], dtype=torch.float32)
            tpos2 = torch.tensor([[
                [-((pos[6] + pos[4]) / 2 - (pos[4] - pos[6]) / downsize_h - ((pos[0] + pos[2]) / 2 - (
                        pos[2] - pos[0]) / downsize_h)) / downsize_w + (pos[6] + pos[4]) / 2 - (pos[4] - pos[6]) / downsize_h,
                 -((pos[7] + pos[5]) / 2 - (pos[5] - pos[7]) / downsize_h - ((pos[1] + pos[3]) / 2 - (
                         pos[3] - pos[1]) / downsize_h)) / downsize_w + (pos[7] + pos[5]) / 2 - (
                         pos[5] - pos[7]) / downsize_h],
                [-((pos[6] + pos[4]) / 2 + (pos[4] - pos[6]) / downsize_h - ((pos[0] + pos[2]) / 2 + (
                        pos[2] - pos[0]) / downsize_h)) / downsize_w + (pos[6] + pos[4]) / 2 + (pos[4] - pos[6]) / downsize_h,
                 -((pos[7] + pos[5]) / 2 + (pos[5] - pos[7]) / downsize_h - ((pos[1] + pos[3]) / 2 + (
                         pos[3] - pos[1]) / downsize_h)) / downsize_w + (pos[7] + pos[5]) / 2 + (
                         pos[5] - pos[7]) / downsize_h],
                [(pos[6] + pos[4]) / 2 + (pos[4] - pos[6]) / downsize_h,
                 (pos[7] + pos[5]) / 2 + (pos[5] - pos[7]) / downsize_h],
                [(pos[6] + pos[4]) / 2 - (pos[4] - pos[6]) / downsize_h,
                 (pos[7] + pos[5]) / 2 - (pos[5] - pos[7]) / downsize_h],
            ]], dtype=torch.float32)
        else:
            downsize_w = 2 * downsize_w
            tpos1 = torch.tensor([[
                [(pos[0] + pos[6]) / 2 - (pos[6] - pos[0]) / downsize_w,
                 (pos[1] + pos[7]) / 2 - (pos[7] - pos[1]) / downsize_w],
                [((pos[2] + pos[4]) / 2 - (pos[4] - pos[2]) / downsize_w - ((pos[0] + pos[6]) / 2 - (
                        pos[6] - pos[0]) / downsize_w)) / downsize_h + (pos[0] + pos[6]) / 2 - (pos[6] - pos[0]) / downsize_w,
                 ((pos[3] + pos[5]) / 2 - (pos[5] - pos[3]) / downsize_w - ((pos[1] + pos[7]) / 2 - (
                         pos[7] - pos[1]) / downsize_w)) / downsize_h + (pos[1] + pos[7]) / 2 - (
                         pos[7] - pos[1]) / downsize_w],
                [((pos[2] + pos[4]) / 2 + (pos[4] - pos[2]) / downsize_w - ((pos[0] + pos[6]) / 2 + (
                        pos[6] - pos[0]) / downsize_w)) / downsize_h + (pos[0] + pos[6]) / 2 + (pos[6] - pos[0]) / downsize_w,
                 ((pos[3] + pos[5]) / 2 + (pos[5] - pos[3]) / downsize_w - ((pos[1] + pos[7]) / 2 + (
                         pos[7] - pos[1]) / downsize_w)) / downsize_h + (pos[1] + pos[7]) / 2 + (
                         pos[7] - pos[1]) / downsize_w],
                [(pos[0] + pos[6]) / 2 + (pos[6] - pos[0]) / downsize_w,
                 (pos[1] + pos[7]) / 2 + (pos[7] - pos[1]) / downsize_w],
            ]], dtype=torch.float32)
            tpos2 = torch.tensor([[
                [-((pos[2] + pos[4]) / 2 - (pos[4] - pos[2]) / downsize_w - ((pos[0] + pos[6]) / 2 - (
                        pos[6] - pos[0]) / downsize_w)) / downsize_h + (pos[2] + pos[4]) / 2 - (pos[4] - pos[2]) / downsize_w,
                 -((pos[3] + pos[5]) / 2 - (pos[5] - pos[3]) / downsize_w - ((pos[1] + pos[7]) / 2 - (
                         pos[7] - pos[1]) / downsize_w)) / downsize_h + (pos[3] + pos[5]) / 2 - (
                         pos[5] - pos[3]) / downsize_w],
                [(pos[2] + pos[4]) / 2 - (pos[4] - pos[2]) / downsize_w,
                 (pos[3] + pos[5]) / 2 - (pos[5] - pos[3]) / downsize_w],
                [(pos[2] + pos[4]) / 2 + (pos[4] - pos[2]) / downsize_w,
                 (pos[3] + pos[5]) / 2 + (pos[5] - pos[3]) / downsize_w],
                [-((pos[2] + pos[4]) / 2 + (pos[4] - pos[2]) / downsize_w - ((pos[0] + pos[6]) / 2 + (
                        pos[6] - pos[0]) / downsize_w)) / downsize_h + (pos[2] + pos[4]) / 2 + (pos[4] - pos[2]) / downsize_w,
                 -((pos[3] + pos[5]) / 2 + (pos[5] - pos[3]) / downsize_w - ((pos[1] + pos[7]) / 2 + (
                         pos[7] - pos[1]) / downsize_w)) / downsize_h + (pos[3] + pos[5]) / 2 + (
                         pos[5] - pos[3]) / downsize_w],
            ]], dtype=torch.float32)
    else:
        if bbox_oneSide > bbox_anotherSide:
            tpos1 = torch.tensor([[
                [(pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize/2,
                 (pos[1] + pos[3]) / 2 - (pos[3] - pos[1]) / downsize/2],
                [(pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize/2,
                 (pos[1] + pos[3]) / 2 + (pos[3] - pos[1]) / downsize/2],
                [((pos[6] + pos[4]) / 2 + (pos[4] - pos[6]) / downsize/2 - ((pos[0] + pos[2]) / 2 + (
                        pos[2] - pos[0]) / downsize/2)) / downsize + (pos[0] + pos[2]) / 2 + (pos[2] - pos[0]) / downsize/2,
                 ((pos[7] + pos[5]) / 2 + (pos[5] - pos[7]) / downsize/2 - ((pos[1] + pos[3]) / 2 + (
                         pos[3] - pos[1]) / downsize/2)) / downsize + (pos[1] + pos[3]) / 2 + (
                         pos[3] - pos[1]) / downsize/2],
                [((pos[6] + pos[4]) / 2 - (pos[4] - pos[6]) / downsize/2 - ((pos[0] + pos[2]) / 2 - (
                        pos[2] - pos[0]) / downsize/2)) / downsize + (pos[0] + pos[2]) / 2 - (pos[2] - pos[0]) / downsize/2,
                 ((pos[7] + pos[5]) / 2 - (pos[5] - pos[7]) / downsize/2 - ((pos[1] + pos[3]) / 2 - (
                         pos[3] - pos[1]) / downsize/2)) / downsize + (pos[1] + pos[3]) / 2 - (
                         pos[3] - pos[1]) / downsize/2],
            ]], dtype=torch.float32)
            tpos2 = torch.tensor([[
                [-((pos[6] + pos[4]) / 2 - (pos[4] - pos[6]) / downsize/2 - ((pos[0] + pos[2]) / 2 - (
                        pos[2] - pos[0]) / downsize/2)) / downsize + (pos[6] + pos[4]) / 2 - (pos[4] - pos[6]) / downsize/2,
                 -((pos[7] + pos[5]) / 2 - (pos[5] - pos[7]) / downsize/2 - ((pos[1] + pos[3]) / 2 - (
                         pos[3] - pos[1]) / downsize/2)) / downsize + (pos[7] + pos[5]) / 2 - (
                         pos[5] - pos[7]) / downsize/2],
                [-((pos[6] + pos[4]) / 2 + (pos[4] - pos[6]) / downsize/2 - ((pos[0] + pos[2]) / 2 + (
                        pos[2] - pos[0]) / downsize/2)) / downsize + (pos[6] + pos[4]) / 2 + (pos[4] - pos[6]) / downsize/2,
                 -((pos[7] + pos[5]) / 2 + (pos[5] - pos[7]) / downsize/2 - ((pos[1] + pos[3]) / 2 + (
                         pos[3] - pos[1]) / downsize/2)) / downsize + (pos[7] + pos[5]) / 2 + (
                         pos[5] - pos[7]) / downsize/2],
                [(pos[6] + pos[4]) / 2 + (pos[4] - pos[6]) / downsize/2,
                 (pos[7] + pos[5]) / 2 + (pos[5] - pos[7]) / downsize/2],
                [(pos[6] + pos[4]) / 2 - (pos[4] - pos[6]) / downsize/2,
                 (pos[7] + pos[5]) / 2 - (pos[5] - pos[7]) / downsize/2],
            ]], dtype=torch.float32)
        else:
            tpos1 = torch.tensor([[
                [(pos[0] + pos[6]) / 2 - (pos[6] - pos[0]) / downsize/2,
                 (pos[1] + pos[7]) / 2 - (pos[7] - pos[1]) / downsize/2],
                [((pos[2] + pos[4]) / 2 - (pos[4] - pos[2]) / downsize/2 - ((pos[0] + pos[6]) / 2 - (
                        pos[6] - pos[0]) / downsize/2)) / downsize + (pos[0] + pos[6]) / 2 - (pos[6] - pos[0]) / downsize/2,
                 ((pos[3] + pos[5]) / 2 - (pos[5] - pos[3]) / downsize/2 - ((pos[1] + pos[7]) / 2 - (
                         pos[7] - pos[1]) / downsize/2)) / downsize + (pos[1] + pos[7]) / 2 - (
                         pos[7] - pos[1]) / downsize/2],
                [((pos[2] + pos[4]) / 2 + (pos[4] - pos[2]) / downsize/2 - ((pos[0] + pos[6]) / 2 + (
                        pos[6] - pos[0]) / downsize/2)) / downsize + (pos[0] + pos[6]) / 2 + (pos[6] - pos[0]) / downsize/2,
                 ((pos[3] + pos[5]) / 2 + (pos[5] - pos[3]) / downsize/2 - ((pos[1] + pos[7]) / 2 + (
                         pos[7] - pos[1]) / downsize/2)) / downsize + (pos[1] + pos[7]) / 2 + (
                         pos[7] - pos[1]) / downsize/2],
                [(pos[0] + pos[6]) / 2 + (pos[6] - pos[0]) / downsize/2,
                 (pos[1] + pos[7]) / 2 + (pos[7] - pos[1]) / downsize/2],
            ]], dtype=torch.float32)
            tpos2 = torch.tensor([[
                [-((pos[2] + pos[4]) / 2 - (pos[4] - pos[2]) / downsize/2 - ((pos[0] + pos[6]) / 2 - (
                        pos[6] - pos[0]) / downsize/2)) / downsize + (pos[2] + pos[4]) / 2 - (pos[4] - pos[2]) / downsize/2,
                 -((pos[3] + pos[5]) / 2 - (pos[5] - pos[3]) / downsize/2 - ((pos[1] + pos[7]) / 2 - (
                         pos[7] - pos[1]) / downsize/2)) / downsize + (pos[3] + pos[5]) / 2 - (
                         pos[5] - pos[3]) / downsize/2],
                [(pos[2] + pos[4]) / 2 - (pos[4] - pos[2]) / downsize/2,
                 (pos[3] + pos[5]) / 2 - (pos[5] - pos[3]) / downsize/2],
                [(pos[2] + pos[4]) / 2 + (pos[4] - pos[2]) / downsize/2,
                 (pos[3] + pos[5]) / 2 + (pos[5] - pos[3]) / downsize/2],
                [-((pos[2] + pos[4]) / 2 + (pos[4] - pos[2]) / downsize/2 - ((pos[0] + pos[6]) / 2 + (
                        pos[6] - pos[0]) / downsize/2)) / downsize + (pos[2] + pos[4]) / 2 + (pos[4] - pos[2]) / downsize/2,
                 -((pos[3] + pos[5]) / 2 + (pos[5] - pos[3]) / downsize/2 - ((pos[1] + pos[7]) / 2 + (
                         pos[7] - pos[1]) / downsize/2)) / downsize + (pos[3] + pos[5]) / 2 + (
                         pos[5] - pos[3]) / downsize/2],
            ]], dtype=torch.float32)

    spos = torch.tensor([[
        [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
    ]], dtype=torch.float32)
    M1 = K.geometry.get_perspective_transform(spos, tpos1).cuda()
    M2 = K.geometry.get_perspective_transform(spos, tpos2).cuda()

    mask1 = K.geometry.warp_perspective(torch.ones_like(pert[0]).unsqueeze(0).float().cuda(), M1,
                                        dsize=(img.shape[-2], img.shape[-1]), mode='nearest')
    mask2 = K.geometry.warp_perspective(torch.ones_like(pert[1]).unsqueeze(0).float().cuda(), M2,
                                        dsize=(img.shape[-2], img.shape[-1]), mode='nearest')

    pert1 = K.geometry.warp_perspective(inv_pert[0].unsqueeze(0).float(), M1, dsize=(img.shape[-2], img.shape[-1]))
    pert2 = K.geometry.warp_perspective(inv_pert[1].unsqueeze(0).float(), M2, dsize=(img.shape[-2], img.shape[-1]))

    inv_perturbed_img = inv_img * (mask1 != 1) * (mask2 != 1) + \
                    pert1 * (mask1 == 1) * (mask2 != 1) + pert2 * (mask2 == 1) * (mask1 != 1)

    perturbed_img = normalize(inv_perturbed_img)

    mask_all = mask1 + mask2

    return perturbed_img, mask_all[:, 0:1]