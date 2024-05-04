# Description: Main file for training and testing the model.
import warnings

warnings.filterwarnings("ignore")
from training import train
from dataset import dataset_generation
from build_model import model_LUT
import argparse
from utils import build_loss, misc
from torch.optim import lr_scheduler
import os
import wandb
import random
import torch
import numpy as np
from functools import partial
from attacking import attack
import logging

logging.getLogger('matplotlib.font_manager').disabled = True


def seed_torch(seed=0):
    print("Seed Fixed!")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# Parse arguments.
def parse_args():
    parser = argparse.ArgumentParser(description='LUT Training and Testing')

    "Global"
    parser.add_argument('--train', action='store_true', help='Whether train LUT model or attack detection model')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the crop region')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the image')
    parser.add_argument('--save_path', type=str, default='./logs', help='Path to save the results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb', action='store_true', help='Use wandb to log the results')

    "LUT part"
    parser.add_argument('--image_root', type=str, default=r'DOTA\all_images',
                        help='Path to the root of images (this will overlap the image_path)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training LUT')
    parser.add_argument('--dataloader_num_workers', type=int, default=12, help='Number of workers for dataloader')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--LUT_dim', type=int, default=7, help='Dimension of the LUT')
    parser.add_argument('--lut_path', type=str, default=r'LUTs',
                        help='Path to the LUT')
    parser.add_argument('--save_visualization', action='store_false', help='Save the visualization results')

    "Attack part"
    parser.add_argument('--save_path_lut', type=str,
                        default=r"\logs\exp\lut\best.pth",
                        help='Path to the trained LUT model')
    parser.add_argument('--attack_image_root', type=str, default=r"attack_images",
                        help='Path to the root of attacked images (this will overlap the image_path)')
    parser.add_argument('--lut_apply', action='store_true', help='Whether to apply LUT on Adv Patch')
    parser.add_argument('--is_eot', action='store_true', help='whether use EOT')
    parser.add_argument('--det_model_config', type=str, default='det_configs/roi_trans_r50_fpn_1x_dota_le90.py',
                        # det_configs/roi_trans_r50_fpn_1x_dota_le90.py
                        # det_configs/oriented_rcnn_r50_fpn_1x_dota_le90.py
                        # det_configs/oriented_reppoints_r50_fpn_1x_dota_le135.py
                        # det_configs/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py
                        help='Detection model config')
    parser.add_argument('--patch_size', type=int, default=40,
                        help='Adversarial patch size. Default setting: Terminal1/2: 40/25, Intersection/City/Street: 10/10/12 Factory: 20')
    parser.add_argument('--attack_epochs', type=int, default=10, help='Attack epochs')
    parser.add_argument('--attack_pos', type=str, default='corner',
                        help='Adv patch position. Default setting: Terminal1/2: corner, Intersection/City/Street/Factory: center',
                        choices=['corner', 'center', 'lateral'])
    parser.add_argument('--tv_weight', type=float, default=100,
                        help='TV loss factor. Default setting Terminal1/2/Factory: 100, Intersection/City/Street: 0')
    parser.add_argument('--nps_weight', type=float, default=0.0, help='NPS loss factor')

    args = parser.parse_args()

    if args.det_model_config == 'det_configs/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py':
        args.det_model_checkpoint = 'pretrained/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth'
    elif args.det_model_config == 'det_configs/roi_trans_r50_fpn_1x_dota_le90.py':
        args.det_model_checkpoint = 'pretrained/roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth'
    elif args.det_model_config == 'det_configs/oriented_rcnn_r50_fpn_1x_dota_le90.py':
        args.det_model_checkpoint = 'pretrained/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
    elif args.det_model_config == 'det_configs/oriented_reppoints_r50_fpn_1x_dota_le135.py':
        args.det_model_checkpoint = 'pretrained/oriented_reppoints_r50_fpn_1x_dota_le135-ef072de9.pth'
    else:
        raise NotImplementedError

    args.save_path = misc.increment_path(os.path.join(args.save_path, "exp1"))
    os.makedirs(args.save_path, exist_ok=True)
    if args.wandb:
        wandb.init(config=args, project="LUT_Attack", name=os.path.basename(args.save_path))
    seed_torch(args.seed)

    logging.basicConfig(filename=os.path.join(args.save_path, 'main.log'), level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    print(args)
    logging.info(args)

    return args


# Training pipline:
def train_pipline(args):
    g = torch.Generator()
    g.manual_seed(args.seed)
    # Generate dataset.
    train_dataset = dataset_generation(args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataloader_num_workers,
                                                   worker_init_fn=partial(worker_init_fn, seed=args.seed), generator=g,
                                                   persistent_workers=True)
    # Generate model.
    model = model_LUT(args).cuda()
    # Define loss function.
    loss_fn = build_loss.loss_generator()
    # Define argsimizer.
    optimizer_params = {
        'lr': args.lr,
        'weight_decay': 1e-2
    }
    optimizer = misc.get_optimizer(model, 'adamw', optimizer_params)

    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.iterations,
                                        pct_start=0.0)
    # Train model.
    train(model, train_dataloader, loss_fn, optimizer, scheduler, args)


# Testing pipline:
def attack_pipline(args):
    # Lut model.
    lut_model = model_LUT(args).cuda()
    lut_model.load_state_dict(torch.load(args.save_path_lut), strict=False)

    # Attack process.
    attack(lut_model, args)


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        args.save_path_lut = os.path.join(args.save_path, "lut")
        os.makedirs(args.save_path_lut, exist_ok=True)
        train_pipline(args)
    else:
        args.save_path_attack = os.path.join(args.save_path, "attack")
        os.makedirs(args.save_path_attack, exist_ok=True)
        attack_pipline(args)
