@echo off
title Train_Digital
G:
cd G:\ComputerPrograms\AISafety\LUT_Attack
call activate pytorch

set attack_image_root=G:/ComputerPrograms/AISafety/LUT_Attack/syn_dataset/render-gen/%1/images

for %%a in (det_configs/roi_trans_r50_fpn_1x_dota_le90.py det_configs/oriented_rcnn_r50_fpn_1x_dota_le90.py det_configs/oriented_reppoints_r50_fpn_1x_dota_le135.py det_configs/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py) do (
	(start python main.py --det_model_config %%a --attack_image_root %attack_image_root%
	ping -n 10 -w 1000 127.0.0.1 > NUL & start python main.py --det_model_config %%a --lut_apply --attack_image_root %attack_image_root%) | pause
	
	(start python main.py --det_model_config %%a --lut_apply --is_eot --attack_image_root %attack_image_root%
	ping -n 10 -w 1000 127.0.0.1 > NUL & start python main.py --det_model_config %%a --is_eot --attack_image_root %attack_image_root%) | pause
)
exit