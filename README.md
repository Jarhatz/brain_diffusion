# Thought2Image-cv-proj
Reconstructing Visual Semantics from Brain Waves

The files that have been heavily modified / newly created are:
- eeg_ldm.py
- ldm_for_eeg.py
- ddpm.py
- cluster_analysis.py
- gen_eval_eeg.py
- eval_generations.py

In order to replicate findings, please make sure all data has been downloaded from their respective links and organized in the mentioned file structure format at the bottom of the page.

Once all of these dependencies have been downloaded and loaded into their correct paths, follow the steps for reproducing the results.

## Environment Setup:
Create and activate conda environment named ```dreamdiffusion``` from the ```environment.yaml```
```sh
conda env create -f environment.yaml
conda activate dreamdiffusion
```

## Download checkpoints

We also checkpoints to run the finetuing and decoding directly.

Project Directory Structure:
```
/pretrains
┣ 📂 models
┃   ┗ 📜 config.yaml
┃   ┗ 📜 v1-5-pruned.ckpt

┣ 📂 generation  
┃   ┗ 📜 checkpoint_best.pth 

┣ 📂 eeg_pretain
┃   ┗ 📜 checkpoint.pth  (pre-trained EEG encoder)

/datasets
┣ 📂 imageNet_images (subset of Imagenet)
┗  📜 imagenet_label_map.csv
┗  📜 block_splits_by_image_all.pth
┗  📜 block_splits_by_image_single.pth 
┗  📜 eeg_5_95_std.pth  

/code
┣ 📂 sc_mbm
┃   ┗ 📜 mae_for_eeg.py
┃   ┗ 📜 trainer.py
┃   ┗ 📜 utils.py

┣ 📂 dc_ldm
┃   ┗ 📜 ldm_for_eeg.py
┃   ┗ 📜 utils.py
┃   ┣ 📂 models
┃   ┃   ┗ (adopted from LDM)
┃   ┣ 📂 modules
┃   ┃   ┗ (adopted from LDM)

┗  📜 stageA1_eeg_pretrain.py   (main script for EEG pre-training)
┗  📜 eeg_ldm.py    (main script for fine-tuning stable diffusion)
┗  📜 gen_eval_eeg.py               (main script for generating images)
┗  📜 eval_generations.py               (main script for evaluating generated images)
┗  📜 cluster_analysis.py                (functions for embedding alignment analysis)
┗  📜 visualize_loss.py                (functions for visualizing losses)
┗  📜 dataset.py                (functions for loading datasets)
┗  📜 eval_metrics.py           (functions for evaluation metrics)
┗  📜 config.py                 (configurations for the main scripts)
```

## Acknowledgement

This code is built upon the publicly available code [DreamDiffusion]([https://github.com/zjc062/mind-vis](https://github.com/bbaaii/DreamDiffusion/tree/main)). Thanks these authors for making their excellent work and codes publicly available.
