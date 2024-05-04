import torch
import os
import tqdm
import wandb
import cv2
import logging


# LUT training pipeline.
def train(model, train_dataloader, loss_fn, optimizer, scheduler, args):
    print("Start training LUT model.")
    logging.info("Start training LUT model.")

    loss_best = 1e10

    # Training.
    model.train()

    acc_loss = 0
    tqdm_bar = tqdm.tqdm(total=args.iterations, desc="Training", position=0)
    for batch_idx, (global_img, crop_aug, crop_mask, crop_gt, lut_reverse, crop_aug_reverse, is_dual_lut) in enumerate(
            train_dataloader):
        # Transfer to GPU.
        global_img = global_img.cuda()
        crop_aug = crop_aug.cuda()
        crop_mask = crop_mask.cuda()
        crop_gt = crop_gt.cuda()
        lut_reverse = lut_reverse.cuda()
        crop_aug_reverse = crop_aug_reverse.cuda()
        is_dual_lut = is_dual_lut.cuda()

        # Forward.
        fit_lut3d, lut_transform_image = model(crop_aug, crop_mask)

        # Compute loss.
        loss_lut_transform_image = loss_fn['masked_mse'](lut_transform_image, crop_gt, crop_mask)
        loss_lut_regularize = loss_fn['regularize_LUT'](fit_lut3d)

        # Since the LUT inversion value is not perfect (usually color inconsistency), we lower the weight of this loss.
        loss_lut_value = loss_fn['mse'](fit_lut3d.permute(0, 2, 3, 4, 1), lut_reverse, is_dual_lut) * 0.1
        loss_tv, loss_mn = loss_fn['TV_3D'](args.LUT_dim)(fit_lut3d)

        # This two loss weight is also lowered to align the magnitude of loss.
        loss_tv *= 0.1
        loss_mn *= 0.1

        loss = loss_lut_transform_image + loss_lut_regularize + loss_lut_value + loss_tv + loss_mn

        acc_loss += loss.item()

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Print Log. (With est time.)
        post_fix = {'acc_loss': acc_loss / (batch_idx + 1), 'loss': loss.item(),
                    'loss_lut_transform_image': loss_lut_transform_image.item(),
                    'loss_lut_regularize': loss_lut_regularize.item() if loss_lut_regularize != 0 else 0,
                    'loss_lut_value': loss_lut_value.item(), 'loss_tv': loss_tv.item(),
                    'loss_mn': loss_mn.item(), 'lr': scheduler.get_last_lr()[0]}
        tqdm_bar.set_postfix(post_fix)
        tqdm_bar.update(1)
        logging.info(post_fix)

        # Remained Time.
        # tqdm_bar.set_postfix_str(tqdm_bar.postfix + " Remained Time: {:.2f}s".format(
        #     tqdm_bar.format_dict['elapsed'] / (batch_idx + 1) * (args.iterations // args.batch_size - batch_idx - 1)))

        if args.wandb:
            # Wandb Log.
            wandb.log({'acc_loss': acc_loss / (batch_idx + 1), "loss": loss.item()})
            wandb.log({"loss_lut_transform_image": loss_lut_transform_image.item()})
            wandb.log({"loss_lut_regularize": loss_lut_regularize.item() if loss_lut_regularize != 0 else 0})
            wandb.log({"loss_lut_value": loss_lut_value.item()})
            wandb.log({"loss_tv": loss_tv.item()})
            wandb.log({"loss_mn": loss_mn.item()})
            wandb.log({"lr": scheduler.get_last_lr()[0]})

        if batch_idx % 50 == 0:
            # Save model.
            if acc_loss / (batch_idx + 1) < loss_best:
                loss_best = acc_loss / (batch_idx + 1)
                torch.save(model.state_dict(), os.path.join(args.save_path_lut, 'best.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(args.save_path_lut, 'last.pth'))

            if args.save_visualization:
                os.makedirs(os.path.join(args.save_path_lut, 'visualization'), exist_ok=True)
                global_img = cv2.cvtColor(global_img[0].div(2.).add(0.5).clamp(0, 1).cpu().numpy().transpose(1, 2, 0),
                                          cv2.COLOR_RGB2BGR)
                crop_aug = cv2.cvtColor(crop_aug[0].div(2.).add(0.5).clamp(0, 1).cpu().numpy().transpose(1, 2, 0),
                                        cv2.COLOR_RGB2BGR)
                crop_mask = crop_mask[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                crop_gt = cv2.cvtColor(crop_gt[0].div(2.).add(0.5).clamp(0, 1).cpu().numpy().transpose(1, 2, 0),
                                       cv2.COLOR_RGB2BGR)
                lut_transform_image = cv2.cvtColor(
                    lut_transform_image[0].detach().div(2.).add(0.5).clamp(0, 1).cpu().numpy().transpose(1, 2, 0),
                    cv2.COLOR_RGB2BGR)
                lut_transform_image = lut_transform_image * crop_mask + crop_gt * (1 - crop_mask)
                crop_aug_reverse = cv2.cvtColor(
                    crop_aug_reverse[0].detach().div(2.).add(0.5).clamp(0, 1).cpu().numpy().transpose(1, 2, 0),
                    cv2.COLOR_RGB2BGR)
                crop_aug_reverse = crop_aug_reverse * crop_mask + crop_gt * (1 - crop_mask)

                cv2.imwrite(os.path.join(args.save_path_lut, 'visualization', '{}_global_img.png'.format(batch_idx)),
                            global_img * 255)
                cv2.imwrite(os.path.join(args.save_path_lut, 'visualization', '{}_crop_aug.png'.format(batch_idx)),
                            crop_aug * 255)
                cv2.imwrite(os.path.join(args.save_path_lut, 'visualization', '{}_crop_mask.png'.format(batch_idx)),
                            crop_mask * 255)
                cv2.imwrite(os.path.join(args.save_path_lut, 'visualization', '{}_crop_gt.png'.format(batch_idx)),
                            crop_gt * 255)
                cv2.imwrite(
                    os.path.join(args.save_path_lut, 'visualization', '{}_lut_transform_image.png'.format(batch_idx)),
                    lut_transform_image * 255)
                cv2.imwrite(
                    os.path.join(args.save_path_lut, 'visualization', '{}_crop_aug_reverse.png'.format(batch_idx)),
                    crop_aug_reverse * 255)

                if args.wandb:
                    # Wandb Visualize.
                    wandb.log({"global_img": wandb.Image(global_img)})
                    wandb.log({"crop_aug": wandb.Image(crop_aug)})
                    wandb.log({"crop_mask": wandb.Image(crop_mask)})
                    wandb.log({"crop_gt": wandb.Image(crop_gt)})
                    wandb.log({"lut_transform_image": wandb.Image(lut_transform_image)})
                    wandb.log({"crop_aug_reverse": wandb.Image(crop_aug_reverse)})

    print("Finish training LUT model.")
    logging.info("Finish training LUT model.")