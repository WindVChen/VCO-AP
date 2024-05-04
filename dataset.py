from torch.utils.data import DataLoader
import torch
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import random
import torchvision.transforms.functional as F
import math
from colour.io.luts.iridas_cube import read_LUT_IridasCube
import logging
import glob


def augment(lut, input_image):
    im_array = np.asarray(input_image, dtype=np.float32) / 255
    is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
    dom_scale = None
    if is_non_default_domain:
        dom_scale = lut.domain[1] - lut.domain[0]
        im_array = im_array * dom_scale + lut.domain[0]

    im_array = lut.apply(im_array)

    if is_non_default_domain:
        im_array = (im_array - lut.domain[0]) / dom_scale
    im_array = im_array * 255
    aug_im = np.uint8(im_array)

    return aug_im


def prepare_mask_and_masked_image(mask):
    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    return mask


# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(int(im_shape[0] * 0.1), int(im_shape[0] * 0.5)),
            random.randint(int(im_shape[0] * 0.1), int(im_shape[1] * 0.5)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask


def free_form_mask(resolution):
    ratio = resolution // 256
    min_num_vertex = 4 * ratio
    max_num_vertex = 12 * ratio
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15
    min_width = 12 * ratio
    max_width = 40 * ratio

    H = W = resolution

    average_radius = math.sqrt(H * H + W * W) / 8
    mask = Image.new('L', (W, H), 0)

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius // 2),
                0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=255, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2,
                          v[1] - width // 2,
                          v[0] + width // 2,
                          v[1] + width // 2),
                         fill=255)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    return mask


class Mask_Compose(transforms.Compose):
    def __call__(self, image, mask=None):
        for t in self.transforms:
            if mask is None:
                image = t(image)
            else:
                image, mask = t(image, mask)
        if mask is None:
            return image
        else:
            return image, mask


class Mask_Resize(transforms.Resize):
    def forward(self, img, mask=None):
        min_resolution = self.size

        scale = random.uniform(0.5, 1.0)
        self.target_size = [int(img.size[1] * scale), int(img.size[0] * scale)]

        # Ensure that the minimum resolution is larger than args.resolution, or will raise error in Mask_RandomCrop().
        if min(self.target_size) < min_resolution:
            scale = min_resolution / min(self.target_size)
            self.target_size = [int(self.target_size[0] * scale + 0.5), int(self.target_size[1] * scale + 0.5)]

        img = F.resize(img, self.target_size, self.interpolation, self.max_size, self.antialias)
        if mask is not None:
            mask = F.resize(mask, self.target_size, F.InterpolationMode.NEAREST, self.max_size, self.antialias)
            return img, mask
        else:
            return img


class Mask_RandomCrop(transforms.RandomCrop):
    def forward(self, img, mask=None):
        i, j, h, w = self.get_params(img, self.size)

        if mask is not None:
            return F.crop(img, i, j, h, w), F.crop(mask, i, j, h, w)
        else:
            return F.crop(img, i, j, h, w)


class Mask_CenterCrop(transforms.CenterCrop):
    def forward(self, img, mask=None):
        if mask is not None:
            return F.center_crop(img, self.size), F.center_crop(mask, self.size)
        else:
            return F.center_crop(img, self.size)


class Mask_RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img, mask=None):
        if mask is None:
            if torch.rand(1) < self.p:
                return F.hflip(img)
            return img
        else:
            if torch.rand(1) < self.p:
                return F.hflip(img), F.hflip(mask)
            return img, mask


# Pytorch dataset preparation.
class dataset_generation(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args

        assert args.image_path is not None or args.image_root is not None, \
            'Either image_path or image_root should be given.'

        self.image_files = [args.image_path]
        if args.image_root is not None:
            self.image_files = sorted(glob.glob(os.path.join(args.image_root, '*')))

        self.pre_transforms = Mask_Compose(
            [
                Mask_Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                Mask_RandomCrop(args.resolution),
                Mask_RandomHorizontalFlip(),
            ]
        )

        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.luts = []
        file_paths = os.walk(os.path.join(args.lut_path, ''))
        for root, dirs, files in file_paths:
            for f in files:
                file_path = os.path.join(root, f)
                if not os.path.isfile(file_path):
                    continue
                elif not file_path.lower().endswith('.cube'):
                    continue
                else:
                    self.luts.append(read_LUT_IridasCube(file_path))
        print('Loaded %d LUTs.' % len(self.luts))
        logging.info('Loaded %d LUTs.' % len(self.luts))

    def __getitem__(self, _):
        img_path = random.choice(self.image_files)
        self.image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        img = self.image.copy()
        img = Image.fromarray(img)

        crop_gt = self.pre_transforms(img)

        mask_generate_type = random.randint(0, 1)
        if mask_generate_type == 0:
            # generate a random mask
            crop_mask = random_mask(crop_gt.size, 1, False)
            # prepare mask and masked image
            crop_mask = prepare_mask_and_masked_image(crop_mask)
        else:
            # use the classic free-form masks in inpainting field
            crop_mask = free_form_mask(crop_gt.size[0])
            crop_mask = prepare_mask_and_masked_image(crop_mask)

        img_resize = transforms.Resize([self.args.resolution, self.args.resolution])(Image.fromarray(self.image))

        # Whether to apply LUT to both Foreground and Background.
        # The threshold is set to not excessively change the color of the background.
        is_dual_lut = (random.uniform(0.0, 1.0) > 0.8)
        if is_dual_lut:
            fore_lut, back_lut = random.sample(self.luts, 2)

            crop_gt = np.array(crop_gt)
            crop_mask = np.array(crop_mask)
            img_resize = np.array(img_resize)

            img_resize = augment(back_lut, img_resize)

            crop_gt_origin = crop_gt.copy()
            crop_gt = augment(back_lut, crop_gt)

            crop_aug = augment(fore_lut, crop_gt_origin)
            crop_aug = crop_aug * np.transpose(crop_mask, (1, 2, 0)) + crop_gt * (
                    1 - np.transpose(crop_mask, (1, 2, 0)))
            crop_aug = crop_aug.astype(np.uint8)

            LUT_reverse = fore_lut.invert(size=self.args.LUT_dim)
            crop_aug_reverse = augment(LUT_reverse, crop_aug)

        else:
            lut = random.sample(self.luts, 1)[0]

            crop_gt = np.array(crop_gt)
            crop_mask = np.array(crop_mask)

            crop_aug = augment(lut, crop_gt)
            crop_aug = crop_aug * np.transpose(crop_mask, (1, 2, 0)) + crop_gt * (
                        1 - np.transpose(crop_mask, (1, 2, 0)))
            crop_aug = crop_aug.astype(np.uint8)

            LUT_reverse = lut.invert(size=self.args.LUT_dim)
            crop_aug_reverse = augment(LUT_reverse, crop_aug)

        return self.train_transforms(img_resize), self.train_transforms(crop_aug), crop_mask, self.train_transforms(
            crop_gt), LUT_reverse.table, self.train_transforms(crop_aug_reverse), is_dual_lut

    def __len__(self):
        return int(self.args.iterations * self.args.batch_size)
