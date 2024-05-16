import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import Config_Generative_Model
from dataset import  create_EEG_dataset
from eeg_ldm import normalize, random_crop, channel_last
from sc_mbm.mae_for_eeg import eeg_encoder,  mapping
from dc_ldm.ldm_for_eeg import eLDM
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

# EXPERIMENTS
# pre: done.
#   python code/cluster_analysis.py -o visuals/pre_eeg
#   Dunn Index: 0.04017865594050018
# subset: cos
#   Dunn Index: 0.07526916117467983
# subset: simclr
#   Dunn Index: 0.07904524260231477
# subset: scl
#   Dunn Index: 0.08089477678963086

# cos: done.
# python code/cluster_analysis.py --checkpoint_path exps/results/generation/cos_300_5e-5/checkpoint_best.pth -o cos_eeg
#       Dunn Index: 0.05509510641954713

# simclr dot: done.
#   python code/cluster_analysis.py --checkpoint_path exps/results/generation/simclr_dot_100_5e-7/checkpoint_best.pth -o simclr_dot_100_eeg
#       Dunn Index: 0.0764004551180696
#   python code/cluster_analysis.py --checkpoint_path exps/results/generation/simclr_dot_100_5e-7/results/generation/simclr_dot_200_5e-7/checkpoint_best.pth -o simclr_dot_200_eeg
#       Dunn Index: 0.06528643688485884

# scl cos: done.
#   python code/cluster_analysis.py --checkpoint_path exps/results/generation/scl_cos_100_5e-7/checkpoint_best.pth -o scl_cos_100_eeg
#       Dunn Index: 0.06703610928052685
#   python code/cluster_analysis.py --checkpoint_path exps/results/generation/11-05-2024-01-32-06/results/generation/11-05-2024-20-39-32/checkpoint_best.pth -o scl_cos_200_eeg
#       Dunn Index: 0.06037135247609568
# scl dot:
#   python ...


def main():
    # project setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,
        transforms.Resize((512, 512)),
        random_crop(config.img_size-crop_pix, p=0.5),
        transforms.Resize((512, 512)),
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize,
        transforms.Resize((512, 512)),
        channel_last
    ])
    if config.dataset == 'EEG':
        eeg_dataset_train, _ = create_EEG_dataset(
            eeg_signals_path=config.eeg_signals_path,
            splits_path = config.splits_path, 
            image_transform=[img_transform_train, img_transform_test],
            subject = config.subject
        )
        num_voxels = eeg_dataset_train.data_len

    print('Loading Pretrained EEG Encoder...')
    metafile = torch.load(config.pretrain_mbm_path, map_location='cpu')
    model = eeg_encoder(
        time_len=num_voxels,
        patch_size=metafile['config'].patch_size,
        embed_dim=metafile['config'].embed_dim,
        depth=metafile['config'].depth,
        num_heads=metafile['config'].num_heads,
        mlp_ratio=metafile['config'].mlp_ratio,
        global_pool=metafile['config'].global_pool
    )
    model.load_checkpoint(metafile['model_state_dict'])
    print('\tDone.\n')
    mapper = mapping()

    if config.checkpoint_path is not None:
        print('Loading Finetuned EEG Encoder...')
        ldm_meta = torch.load(config.checkpoint_path, map_location='cpu')
        diffusion_model = eLDM(metafile, num_voxels, device=device, pretrain_root=config.pretrain_gm_path, logger=None, 
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond, clip_tune=config.clip_tune, cls_tune=config.cls_tune)
        diffusion_model.model.load_state_dict(ldm_meta['model_state_dict'])
        model = diffusion_model.model.cond_stage_model.mae
        print('\tDone.\n')


    train_loader = DataLoader(eeg_dataset_train, batch_size=8, num_workers=12, shuffle=True)

    eeg_embs = None
    labels = None
    for idx, batch in enumerate(train_loader):
        eegs = batch['eeg']
        labs = batch['label']
        embs = model(eegs)
        embs = mapper(embs)
        if idx == 0:
            eeg_embs = embs
            labels = labs
        else:
            eeg_embs = torch.cat([eeg_embs, embs], dim=0)
            labels = torch.cat([labels, labs], dim=0)
    
    eeg_embs = eeg_embs.detach().numpy()
    labels = labels.numpy()
    unique_labels = np.unique(labels)

    # t-SNE Dimensional Reduction
    print('Starting t-SNE Feature Reduction...')
    print('Before: ', eeg_embs.shape)
    eeg_tsne = TSNE(n_components=2, perplexity=30).fit_transform(eeg_embs)
    print('After: ', eeg_tsne.shape)
    print('\tDone.\n')

    if args.output_file:
        plot_scatter(eeg_tsne, labels, unique_labels)

    # Preprocess for Dunn Index
    clusters = []
    for i, label in enumerate(unique_labels):
        indices = labels == label
        clusters.append(eeg_tsne[indices, 0])
    di = dunn(clusters)
    print(f'Dunn Index: {di}')
    # print(f'DI (log-scale): {log_di}')


def dunn(clusters):
    δs = np.ones([len(clusters), len(clusters)])
    Δs = np.zeros([len(clusters), 1])
    j_range = list(range(0, len(clusters)))
    for i in j_range:
        for l in (j_range[0:i] + j_range[i+1:]):
            δs[i, l] = δ(clusters[i], clusters[l])
            Δs[i] = Δ(clusters[i])
    di = np.min(δs)/np.max(Δs)
    return di

# Calculates distance between two clusters
def δ(ci, cj):
    values = np.ones([len(ci), len(cj)])
    for i in range(0, len(ci)):
        for j in range(0, len(cj)):
            values[i, j] = np.linalg.norm(ci[i] - cj[j])
    return np.mean(values)

# Calculates size/spread of a cluster
def Δ(ck):
    values = np.zeros([len(ck), len(ck)])
    for i in range(0, len(ck)):
        for j in range(0, len(ck)):
            values[i, j] = np.linalg.norm(ck[i] - ck[j])
    return np.mean(values)


def plot_scatter(points, labels, unique_labels):
    label_map = pd.read_csv('datasets/imagenet_label_map.csv')
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(16, 12))

    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(points[indices, 0], points[indices, 1], marker='o', c=colors[i], label=label_map.iloc[label]['annotation'])

    plt.legend(title='Class', loc='upper right') # bbox_to_anchor=(1, 1)
    plt.title('t-SNE Reduced EEG Embedding Space')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.grid(True)
    plt.savefig(os.path.join('visuals', f'{args.output_file}.png'))


def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    parser.add_argument('--root_path', type=str, default = '.')
    parser.add_argument('--pretrain_mbm_path', type=str, default='pretrains/eeg_pretrain/checkpoint.pth')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('-o', '--output_file', type=str)
    return parser

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_Generative_Model()
    config = update_config(args, config)
    main()