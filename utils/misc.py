import re
from pathlib import Path
import glob
import logging
import numpy as np
import torch
import math
from adamp import AdamP
import random
import torch.nn as nn

_logger = None


def custom_normalize(x, mode='normal', typ=False):
    mean = torch.tensor(np.array([123.675, 116.28, 103.53]), dtype=x.dtype)[np.newaxis, :, np.newaxis,
           np.newaxis]
    var = torch.tensor(np.array([58.395, 57.12, 57.375]), dtype=x.dtype)[np.newaxis, :, np.newaxis, np.newaxis]
    mean = mean.to(x.device)
    var = var.to(x.device)
    if typ:
        mean = mean.half()
        var = var.half()
    if mode == 'normal':
        return (x - mean) / var
    elif mode == 'inv':
        return x * var + mean


def increment_path(path):
    # Increment path, i.e. runs/exp1 --> runs/exp{sep}1, runs/exp{sep}2 etc.
    res = re.search("\d+", path)
    if res is None:
        print("Set initial exp number!")
        exit(1)

    if not Path(path).exists():
        return str(path)
    else:
        path = path[:res.start()]
        dirs = glob.glob(f"{path}*")  # similar paths
        matches = [re.search(rf"%s(\d+)" % Path(path).stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1  # increment number
        return f"{path}{n}"  # update path


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, fmt=':f'):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_logger(log_file, level=logging.INFO):
    global _logger
    _logger = logging.getLogger()
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    _logger.setLevel(level)
    _logger.addHandler(fh)
    _logger.addHandler(sh)

    return _logger


def normalize(x, opt, mode='normal'):
    device = x.device
    mean = torch.tensor(np.array([.5, .5, .5]), dtype=x.dtype)[np.newaxis, :, np.newaxis, np.newaxis].to(device)
    var = torch.tensor(np.array([.5, .5, .5]), dtype=x.dtype)[np.newaxis, :, np.newaxis, np.newaxis].to(device)
    if mode == 'normal':
        return (x - mean) / var
    elif mode == 'inv':
        return x * var + mean


def get_optimizer(model, opt_name, opt_kwargs):
    params = []
    base_lr = opt_kwargs['lr']
    for name, param in model.named_parameters():
        param_group = {'params': [param]}
        if not param.requires_grad:
            params.append(param_group)
            continue

        if not math.isclose(getattr(param, 'lr_mult', 1.0), 1.0):
            # print(f'Applied lr_mult={param.lr_mult} to "{name}" parameter.')
            param_group['lr'] = param_group.get('lr', base_lr) * param.lr_mult

        params.append(param_group)

    optimizer = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'adamp': AdamP
    }[opt_name.lower()](params, **opt_kwargs)

    return optimizer


class LRMult(object):
    def __init__(self, lr_mult=1.):
        self.lr_mult = lr_mult

    def __call__(self, m):
        if getattr(m, 'weight', None) is not None:
            m.weight.lr_mult = self.lr_mult
        if getattr(m, 'bias', None) is not None:
            m.bias.lr_mult = self.lr_mult


def customRandomCrop(objects, crop_height, crop_width, h_start=None, w_start=None):
    if h_start is None:
        h_start = random.random()
    if w_start is None:
        w_start = random.random()
    if isinstance(objects, list):
        out = []
        for obj in objects:
            out.append(random_crop(obj, crop_height, crop_width, h_start, w_start))

    else:
        out = random_crop(objects, crop_height, crop_width, h_start, w_start)

    return out, h_start, w_start


def get_random_crop_coords(height: int, width: int, crop_height: int, crop_width: int, h_start: float,
                           w_start: float):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(img: np.ndarray, crop_height: int, crop_width: int, h_start: float, w_start: float):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img


class PadToDivisor:
    def __init__(self, divisor):
        super().__init__()
        self.divisor = divisor

    def transform(self, images):

        self._pads = (*self._get_dim_padding(images[0].shape[-1]), *self._get_dim_padding(images[0].shape[-2]))
        self.pad_operation = nn.ZeroPad2d(padding=self._pads)

        out = []
        for im in images:
            out.append(self.pad_operation(im))

        return out

    def inv_transform(self, image):
        assert self._pads is not None,\
            'Something went wrong, inv_transform(...) should be called after transform(...)'
        return self._remove_padding(image)

    def _get_dim_padding(self, dim_size):
        pad = (self.divisor - dim_size % self.divisor) % self.divisor
        pad_upper = pad // 2
        pad_lower = pad - pad_upper

        return pad_upper, pad_lower

    def _remove_padding(self, tensors):
        tensor_h, tensor_w = tensors[0].shape[-2:]
        out = []
        for t in tensors:
            out.append(t[..., self._pads[2]:tensor_h - self._pads[3], self._pads[0]:tensor_w - self._pads[1]])
        return out
