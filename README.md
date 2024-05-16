# Thought2Image-cv-proj
Reconstructing Visual Semantics from Brain Waves

The files that have been heavily modified / newly created are:
```
code/sc_mbm/mae_for_eeg.py

code/eeg_ldm.py

code/dc_ldm/ldm_for_eeg.py

code/dc_ldm/models/diffusion/ddpm.py

code/cluster_analysis.py

code/gen_eval_eeg.py

code/eval_generations.py
```

In order to replicate findings, please make sure all data has been downloaded from their respective links and organized in the mentioned file structure format at the bottom of the page.

- [EEG Waves](https://github.com/perceivelab/eeg_visual_classification) : Please download and place them in `/datasets` and `/pretrains` folders in the project root dir
- [Image Pairs](https://drive.google.com/file/d/1y7I9bG1zKYqBM94odcox_eQjnP9HGo9-/view?usp=drive_link) : Please download ImageNet subset of shown images
- [Stable Diffusion 1.5 Checkpoint](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt?download=true) : Please download SD 1.5 checkpoint and place in `/pretrains/models`

Once all of these dependencies have been downloaded and loaded into their correct paths, follow the steps for reproducing the results.

## Environment Setup:
Create and activate conda environment named ```dreamdiffusion``` from the ```environment.yaml```
```sh
conda env create -f environment.yaml
conda activate dreamdiffusion
```

## Finetuning the Stable Diffusion with Pre-trained EEG Encoder:
In this stage, the cross-attention heads and pre-trained EEG encoder will be jointly optimized with EEG-image pairs. 

```sh
python code/eeg_ldm.py --dataset EEG --batch_size 10 --num_epoch 100 --lr 1e-5
```
Optionally you can also provide a checkpoint file to resume from:
```sh
python code/eeg_ldm.py --dataset EEG --batch_size 10 --num_epoch 100 --lr 1e-5 --checkpoint_path [CHECKPOINT_PATH]
```

## Cluster Analysis:
After fine-tuning, EEG Encoder's embedding space can be visually plotted with t-SNE dimensional reduction. 

```sh
python code/cluster_analysis.py --checkpoint_path [CHECKPOINT_PATH] -o [OUTPUT_PATH]
```

## Visualizing Loss:
Plot loss curves. 

```sh
python code/visualize_loss.py --loss_path [LOSS_PATH]
```

## Generating Images:
Generate images based on the held out test EEG dataset.

```sh
python code/gen_eval_eeg.py --dataset EEG --model_path [MODEL_PATH]
```

## Evaluate Generated Images:
Run evaluation metrics on the generated images from the test set. ViT ImageNet-1K classifier used for Top1-Acc and Top3-Acc scores.

```sh
python code/eval_generations.py --results_path [RESULTS_PATH]
```

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
