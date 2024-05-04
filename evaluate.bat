@echo off
title Evaluate_Physical
G:
cd G:\ComputerPrograms\AISafety\LUT_Attack
call activate pytorch

call python main.py --attack_image_root G:/ComputerPrograms/AISafety/LUT_Attack/syn_dataset/render-gen/%1/lut/images --attack_epochs 0 --det_model_config %2
exit