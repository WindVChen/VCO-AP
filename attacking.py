import glob

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
import torch
import os
import numpy as np
import cv2
from utils.misc import custom_normalize as normalize
import art.preprocessing.expectation_over_transformation as EOT
from utils import transforms_rbbox
from utils.adversarial_attack_pos import addPert_corner, addPert_lateral, addPert_center
from collections import OrderedDict
import shutil
from mmrotate.models import build_detector
from mmrotate.core.bbox.transforms import obb2poly_le90, obb2poly_le135, obb2poly_oc
from mmrotate.utils import build_dp
import wandb
import logging
from mmdet.apis import single_gpu_test
from PIL import Image
from torchvision import transforms
from utils.build_loss import loss_generator
from mmrotate.apis import inference_detector_by_patches
from mmdet.apis import show_result_pyplot
import re
import random


def prepare_and_split_img(args):
    split_list = []
    if args.attack_image_root is not None:
        for img in os.listdir(args.attack_image_root):
            split_list.append(os.path.join(args.attack_image_root, img))
    else:
        split_list.append(args.image_path)

    # Image prepare.
    origin_image_dir = os.path.join(args.save_path_attack, 'origin_data', "image")
    os.makedirs(origin_image_dir, exist_ok=True)
    origin_label_dir = os.path.join(args.save_path_attack, 'origin_data', "labelTxt")
    os.makedirs(origin_label_dir, exist_ok=True)

    for image_path in split_list:
        shutil.copy(image_path, origin_image_dir)

        # This is for possible creation of the prediction labelTxt file.
        if not os.path.exists(image_path.replace("images", "labelTxt").replace("png", "txt")):
            with open(os.path.join(origin_label_dir,
                                   os.path.basename(image_path.replace("images", "labelTxt").replace("png", "txt"))),
                      'w') as f:
                f.write("")
        else:
            shutil.copy(image_path.replace("images", "labelTxt").replace("png", "txt"), origin_label_dir)

        "Remove the first lines of the labelTxt file."
        with open(os.path.join(origin_label_dir, os.path.basename(image_path.replace("images", "labelTxt").replace("png", "txt"))), 'r') as f:
            lines = f.readlines()
            start = 0
            for i in range(len(lines)):
                items = lines[i].split(' ')
                if len(items) >= 9:
                    break
                start += 1
        with open(os.path.join(origin_label_dir, os.path.basename(image_path.replace("images", "labelTxt").replace("png", "txt"))), 'w') as f:
            f.writelines(lines[start:])

    # DOTA image split.
    print("Splitting DOTA image...")
    logging.info("Splitting DOTA image...")
    split_root = os.path.join(args.save_path_attack, 'split_data')
    # os.makedirs(split_root, exist_ok=True)
    command = f"python utils/img_split.py --nproc 1 --img-dirs {origin_image_dir} --ann-dirs {origin_label_dir} " \
              f"--sizes {args.resolution} --gaps {args.resolution // 2} --padding-value 104 116 124 " \
              f"--save-dir {split_root}"
    os.system(command)
    print("Finish DOTA image split.\n")
    logging.info("Finish DOTA image split.\n")

    return split_root


def save_clean_imgs(config, args):
    # Assure that there is no augmentation applied to original image (for correct evaluation with gt labels)
    config.train_pipeline[3]['flip_ratio'] = [0., 0., 0.]
    config.data['train']['pipeline'] = config.train_pipeline
    train_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    train_loader_cfg = {
        **train_dataloader_default_args,
        **config.data.get('train_dataloader', {}),
        "persistent_workers": True
    }
    dataset = build_dataset(config.data.train, default_args={"filter_empty_gt": False})
    data_loader = build_dataloader(dataset, **train_loader_cfg)

    # Save the clean images (only those with targets).
    print("Start to save clean images (only those with targets)!\n")
    logging.info("Start to save clean images (only those with targets)!\n")
    with torch.no_grad():
        for batch_i, data in enumerate(data_loader):
            path = data['img_metas'].data[0]
            tmpData = data['img'].data[0].clone().cuda()

            # eot applied
            tmpData = normalize(tmpData, 'inv').clamp(0., 255.)

            # Save the perturbed images and the perturbation patch.
            adv_img = tmpData.permute(0, 2, 3, 1).cpu().squeeze().numpy().astype(np.int64).astype(np.uint8)
            out_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)

            os.makedirs(os.path.join(args.save_path_attack, "clean_images"), exist_ok=True)
            cv2.imwrite(os.path.join(args.save_path_attack, "clean_images", os.path.basename(path[0]['filename'])),
                        out_img)


def save_perturbations_and_perturbed_imgs(lut_model, global_images, args, attack_patch, config, patch_num, all_classes,
                                          no_lut=False):
    # Assure that there is no augmentation applied to original image (for correct evaluation with gt labels)
    config.train_pipeline[3]['flip_ratio'] = [0., 0., 0.]
    config.data['train']['pipeline'] = config.train_pipeline
    train_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    train_loader_cfg = {
        **train_dataloader_default_args,
        **config.data.get('train_dataloader', {}),
        "persistent_workers": True
    }
    dataset = build_dataset(config.data.train, default_args={"filter_empty_gt": False})
    data_loader = build_dataloader(dataset, **train_loader_cfg)

    # Save the perturbed images.
    print("Start to save perturbations and perturbed images!\n")
    logging.info("Start to save perturbations and perturbed images!\n")
    with torch.no_grad():
        for batch_i, data in enumerate(data_loader):
            path = data['img_metas'].data[0]
            tmpData = data['img'].data[0].clone().cuda()

            patch = normalize(attack_patch * 255)
            pos = data['gt_bboxes'].data[0][0]

            if config.angle_version == 'oc':
                pos = obb2poly_oc(pos)
            elif config.angle_version == 'le135':
                pos = obb2poly_le135(pos)
            elif config.angle_version == 'le90':
                pos = obb2poly_le90(pos)
            else:
                raise NotImplementedError

            all_mask = None
            for j in range(len(data['gt_labels'].data[0][0])):
                if transforms_rbbox.if_zero_area(pos[j]):
                    continue

                # Apply the adv_patches onto the image.
                if args.attack_pos == "corner":
                    tmpData, mask = addPert_corner(
                        patch[data['gt_labels'].data[0][0][j] * patch_num:data['gt_labels'].data[0][0][
                                                                              j] * patch_num + patch_num], tmpData,
                        pos[j])
                elif args.attack_pos == "lateral":
                    tmpData, mask = addPert_lateral(
                        patch[data['gt_labels'].data[0][0][j] * patch_num:data['gt_labels'].data[0][0][
                                                                              j] * patch_num + patch_num], tmpData,
                        pos[j])
                elif args.attack_pos == "center":
                    tmpData, mask = addPert_center(
                        patch[data['gt_labels'].data[0][0][j] * patch_num:data['gt_labels'].data[0][0][
                                                                              j] * patch_num + patch_num], tmpData,
                        pos[j])

                if all_mask is not None:
                    all_mask = (all_mask + mask).clamp(0., 1.)
                else:
                    all_mask = mask

            mask = all_mask

            # eot applied
            tmpData = normalize(tmpData, 'inv').clamp(0., 255.)

            if mask is not None and args.lut_apply and not no_lut:
                # Apply harmonization on Adv patch
                tmp_tmpData = tmpData.detach().clone()
                tmpData = (tmpData / 255. - 0.5) / 0.5

                _, tmpData = lut_model(tmpData, mask)
                tmpData = ((tmpData * 0.5 + 0.5) * 255.).clamp(0., 255.)
                tmpData = tmpData * mask + tmp_tmpData * (1 - mask)

            # Save the perturbed images and the perturbation patch.
            adv_img = tmpData.permute(0, 2, 3, 1).cpu().squeeze().numpy().astype(np.int64).astype(np.uint8)
            out_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)

            patch = torch.clip(attack_patch.cpu() * 255, 0., 255.).permute(0, 2, 3, 1).numpy().astype(np.int64).astype(
                np.uint8)

            os.makedirs(os.path.join(args.save_path_attack, "attack_images" if not no_lut else "attack_images_noLut"),
                        exist_ok=True)
            cv2.imwrite(os.path.join(args.save_path_attack, "attack_images" if not no_lut else "attack_images_noLut",
                                     os.path.basename(path[0]['filename'])),
                        out_img)

        os.makedirs(os.path.join(args.save_path_attack, "adv_patches"), exist_ok=True)
        for cate in all_classes:
            for num_p in range(patch_num):
                out_noise = cv2.cvtColor(patch[cate * patch_num + num_p], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.save_path_attack, "adv_patches",
                                         'perturbation_{}_{}.png'.format(cate, num_p)), out_noise)

    "Save merged full perturbed image."
    print("Start to save merged full perturbed image!\n")
    logging.info("Start to save merged full perturbed image!\n")

    merge_list = []
    if args.attack_image_root is not None:
        for img in os.listdir(args.attack_image_root):
            merge_list.append(os.path.join(args.attack_image_root, img))
    else:
        merge_list.append(args.image_path)

    for image_path in merge_list:
        original_image = cv2.imread(image_path)
        for im in glob.glob(
                os.path.join(args.save_path_attack, "attack_images" if not no_lut else "attack_images_noLut", "*")):
            if os.path.basename(image_path)[:-4] not in im:
                continue
            x, y = re.findall("\d+", os.path.basename(im))[-2:]
            x, y = int(x), int(y)
            patch = cv2.imread(im)
            original_image[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
        os.makedirs(
            os.path.join(args.save_path_attack, "attack_image_merge" if not no_lut else "attack_image_merge_noLut"),
            exist_ok=True)
        cv2.imwrite(
            os.path.join(args.save_path_attack, "attack_image_merge" if not no_lut else "attack_image_merge_noLut",
                         os.path.basename(image_path)), original_image)


def evaluate(config, args, mode):
    print("Evaluate the {} images ...\n".format(mode))
    logging.info("Evaluate the {} images ...\n".format(mode))
    config.data['test']['img_prefix'] = \
        os.path.join(args.save_path_attack, "clean_images" if mode == "clean" else "attack_images")
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    config.data.test.test_mode = True
    test_loader_cfg = {
        **test_dataloader_default_args,
        **config.data.get('test_dataloader', {})
    }
    dataset = build_dataset(config.data.test, default_args={"filter_empty_gt": False})
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.det_model_checkpoint, map_location='cuda:0')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model = build_dp(model, device_ids=[0])
    os.makedirs(os.path.join(args.save_path_attack,
                             "clean_image_outputs" if mode == "clean" else "attack_image_outputs"), exist_ok=True)
    outputs = single_gpu_test(
        model, data_loader, False,
        os.path.join(args.save_path_attack, "clean_image_outputs" if mode == "clean" else "attack_image_outputs"), 0.3)
    metric = dataset.evaluate(outputs, logger=logging.root)
    print(metric)
    logging.info(metric)


def detection(args, mode):
    "This is for detecting the full large image and save the detection results."
    print("Detect the {} images ...\n".format(mode))
    logging.info("Detect the {} images ...\n".format(mode))
    device = 'cuda:0'
    config = mmcv.Config.fromfile(args.det_model_config)
    config.model.pretrained = None
    config.test_pipeline[1]['img_scale'] = (args.resolution, args.resolution)
    config.data['test']['pipeline'] = config.test_pipeline
    model = build_detector(config.model)
    checkpoint = load_checkpoint(model, args.det_model_checkpoint, map_location=device)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config
    model.to(device)
    model.eval()

    det_list = []
    if args.attack_image_root is not None:
        for img in os.listdir(args.attack_image_root):
            det_list.append(os.path.join(args.attack_image_root, img))
    else:
        det_list.append(args.image_path)

    for image_path in det_list:
        img = image_path if mode == "clean" else os.path.join(args.save_path_attack, 'attack_image_merge',
                                                              os.path.basename(image_path))
        result = inference_detector_by_patches(model, img, [args.resolution], [args.resolution // 2], [1.0], 0.1)

        "Save image with detection results"
        origin_image_dir = os.path.join(args.save_path_attack,
                                        'clean_image_merge_output' if mode == "clean" else 'attack_image_merge_output',
                                        "image")
        os.makedirs(origin_image_dir, exist_ok=True)

        # result = [result[9]]
        show_result_pyplot(model, img, result, score_thr=0.3, palette='dota',
                           out_file=os.path.join(origin_image_dir, os.path.basename(img)))

        txt = open(os.path.join(args.save_path_attack,
                                'clean_image_merge_output' if mode == "clean" else 'attack_image_merge_output',
                                os.path.basename(img)[:-4] + '.txt'), 'w')
        bboxes = np.vstack(result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)
        scores = bboxes[:, -1]
        inds = scores > 0.3
        bboxes = bboxes[inds, :]
        labels = labels[inds]

        for i, bbox in enumerate(bboxes):
            xc, yc, w, h, ag = bbox[:5]
            wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
            hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
            p1 = (xc - wx - hx, yc - wy - hy)
            p2 = (xc + wx - hx, yc + wy - hy)
            p3 = (xc + wx + hx, yc + wy + hy)
            p4 = (xc - wx + hx, yc - wy + hy)
            poly = np.int0(np.array([p1, p2, p3, p4]))

            txt.write(' '.join([str(int(x)) for x in poly.reshape(-1)]) + ' ' + model.CLASSES[labels[i]] + ' 0' + '\n')
        txt.close()


def parse_losses(losses):
    "Sum up each loss, return cls loss"
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum((-0 if 'cls' not in _key else -_value) for _key, _value in log_vars.items() if 'loss' in _key)
    return loss


# Attack pipeline.
def attack(lut_model, args):
    print("Start attacking detection model.")
    logging.info("Start attacking detection model.")

    # Split input large images into patches.
    split_root = prepare_and_split_img(args)

    # Init global image.
    global_images = []
    if args.attack_image_root is not None:
        for img in os.listdir(args.attack_image_root):
            global_images.append(os.path.join(args.attack_image_root, img))
    else:
        global_images.append(args.image_path)

    # Choose to use a config and initialize the detector
    config = args.det_model_config

    # Setup a checkpoint file to load
    checkpoint = args.det_model_checkpoint

    # Set the device to be used for evaluation
    device = 'cuda:0'

    # Load the config
    config = mmcv.Config.fromfile(config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Build dataset
    config.data_root = split_root
    config.train_pipeline[2]['img_scale'] = (args.resolution, args.resolution)
    config.data['train']['ann_file'] = os.path.join(config.data_root, 'annfiles/')
    config.data['train']['img_prefix'] = os.path.join(config.data_root, 'images/')
    config.data['train']['pipeline'] = config.train_pipeline

    config.test_pipeline[1]['img_scale'] = (args.resolution, args.resolution)
    config.data['test']['ann_file'] = os.path.join(config.data_root, 'annfiles/')
    config.data['test']['pipeline'] = config.test_pipeline

    train_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    train_loader_cfg = {
        **train_dataloader_default_args,
        **config.data.get('train_dataloader', {}),
        "persistent_workers": True
    }
    dataset = build_dataset(config.data.train, default_args={"filter_empty_gt": False})
    data_loader = build_dataloader(dataset, **train_loader_cfg)

    # Initialize the detector
    model = build_detector(config.model, train_cfg=config.get('train_cfg'), test_cfg=config.get('test_cfg'))

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to MMDataParallel
    model = build_dp(model, device_ids=[0])
    # Convert the model into evaluation mode
    model.eval()
    lut_model.eval()

    model.requires_grad_(False)
    lut_model.requires_grad_(False)
    lut_model = lut_model.cuda()

    # Initialize Adversarial Patch
    if args.attack_pos == 'corner':
        patch_num = 4
    elif args.attack_pos == 'center':
        patch_num = 1
    elif args.attack_pos == 'lateral':
        patch_num = 2
    else:
        raise NotImplementedError

    attack_patch = torch.nn.Parameter(
        torch.rand(len(model.module.CLASSES) * patch_num, 3, args.patch_size, args.patch_size).cuda())
    optimizer = torch.optim.Adam([attack_patch], lr=0.03)
    optimizer.zero_grad()

    # Initialize EOT
    eot_sequence = [
        # EoTImageRotationPyTorch(nb_samples=1, angles=90.0, clip_values=(0., 255.), label_type="classification"),
        EOT.EoTBrightnessPyTorch(nb_samples=1, delta=0.1 * 255, clip_values=(0., 255.)),
        EOT.EoTContrastPyTorch(nb_samples=1, contrast_factor=(0.8, 1.2), clip_values=(0., 255.)),
        # EOT.EoTImageCenterCropPyTorch(nb_samples=1, size=50, clip_values=(0., 255.)),
    ]

    tv_inference = loss_generator()['TVLoss']()
    nps_inference = loss_generator()['NPSCalculator'](
        "./pretrained/30values.txt",
        attack_patch.shape[-1])
    lossavg = []
    lossavg_det = []
    lossavg_tv = []
    lossavg_nps = []

    all_classes = []
    # Store all the related classes.
    for batch_i, data in enumerate(data_loader):
        for j in range(len(data['gt_labels'].data[0][0])):
            all_classes.append(data['gt_labels'].data[0][0][j])

    # Attack loop.
    for epo in range(args.attack_epochs):
        for batch_i, data in enumerate(data_loader):
            tmpData = data['img'].data[0].clone().cuda()

            patch = normalize(attack_patch * 255)
            pos = data['gt_bboxes'].data[0][0]

            if config.angle_version == 'oc':
                pos = obb2poly_oc(pos)
            elif config.angle_version == 'le135':
                pos = obb2poly_le135(pos)
            elif config.angle_version == 'le90':
                pos = obb2poly_le90(pos)
            else:
                raise NotImplementedError

            all_mask = None
            for j in range(len(data['gt_labels'].data[0][0])):
                if transforms_rbbox.if_zero_area(pos[j]):
                    continue

                if args.attack_pos == "corner":
                    tmpData, mask = addPert_corner(
                        patch[
                        data['gt_labels'].data[0][0][j] * patch_num:data['gt_labels'].data[0][0][
                                                                        j] * patch_num + patch_num],
                        tmpData, pos[j])
                elif args.attack_pos == "lateral":
                    tmpData, mask = addPert_lateral(
                        patch[
                        data['gt_labels'].data[0][0][j] * patch_num:data['gt_labels'].data[0][0][
                                                                        j] * patch_num + patch_num],
                        tmpData, pos[j])
                elif args.attack_pos == "center":
                    tmpData, mask = addPert_center(
                        patch[
                        data['gt_labels'].data[0][0][j] * patch_num:data['gt_labels'].data[0][0][
                                                                        j] * patch_num + patch_num],
                        tmpData, pos[j])

                if all_mask is not None:
                    all_mask = (all_mask + mask).clamp(0., 1.)
                else:
                    all_mask = mask

            mask = all_mask
            # eot applied
            tmpData = normalize(tmpData, 'inv').clamp(0., 255.)

            if mask is not None and args.lut_apply and random.uniform(0, 1) < 0.5:
                # Apply harmonization on Adv patch
                tmp_tmpData = tmpData.detach().clone()
                tmpData = (tmpData / 255. - 0.5) / 0.5

                _, tmpData = lut_model(tmpData, mask)
                tmpData = ((tmpData * 0.5 + 0.5) * 255.).clamp(0., 255.)
                tmpData = tmpData * mask + tmp_tmpData * (1 - mask)

            if args.is_eot:
                tmpData = tmpData.permute(0, 2, 3, 1)
                for trans in eot_sequence:
                    tmpData = trans.forward(tmpData, None)[0]
                tmpData = tmpData.permute(0, 3, 1, 2)
            tmpData = normalize(tmpData)
            tmpData = tmpData.contiguous()

            data['img'].data[0] = tmpData

            try:
                result = model(return_loss=True, **data)
            except:
                # print(f"Error in forward pass, epo: {epo}, batch: {batch_i}, continue")
                continue

            loss_det = parse_losses(result)
            if loss_det.grad_fn == None:
                continue
            loss_tv = tv_inference(attack_patch) * args.tv_weight
            loss_nps = nps_inference(attack_patch) * args.nps_weight

            lossavg_det.append(loss_det.item())
            lossavg_tv.append(loss_tv.item())
            lossavg_nps.append(loss_nps.item())

            loss = loss_det + loss_tv + loss_nps
            loss.backward()

            lossavg.append(loss.item())
            # Accumulate gradients
            if (batch_i + 1) % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    attack_patch[:] = torch.clamp(attack_patch, min=0., max=1.)

        log_out = "time: {}\tmemory: {}\tepoch: {}\tloss: {}\tlossDet: {}\tlossTV: {}\tlossNPS: {}".format(
            transforms_rbbox.get_time_str(), torch.cuda.memory_allocated(0) // 1024 ** 2, epo, np.array(lossavg).mean(),
            np.array(lossavg_det).mean(), np.array(lossavg_tv).mean(), np.array(lossavg_nps).mean())
        if args.wandb:
            # Wandb Log.
            wandb.log({'loss': np.array(lossavg).mean(), 'lossDet': np.array(lossavg_det).mean(),
                       'lossTV': np.array(lossavg_tv).mean(), 'lossNPS': np.array(lossavg_nps).mean()})
        print(log_out)
        logging.info(log_out)

        torch.save(attack_patch.cpu().detach(), os.path.join(args.save_path_attack, "perturbations{}.pth".format(epo)))
        print("Save epoch {} perturbations!".format(epo))
        logging.info("Save epoch {} perturbations!".format(epo))

    save_clean_imgs(config, args)

    save_perturbations_and_perturbed_imgs(lut_model, global_images, args, attack_patch, config, patch_num, all_classes)
    save_perturbations_and_perturbed_imgs(lut_model, global_images, args, attack_patch, config, patch_num, all_classes,
                                          no_lut=True)

    evaluate(config, args, "clean")
    evaluate(config, args, "perturbed")

    detection(args, "clean")
    detection(args, "perturbed")

    print("Finish attacking detection model.")
    logging.info("Finish attacking detection model.")
