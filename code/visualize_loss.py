import os, argparse
import numpy as np
from matplotlib import pyplot as plt


def plot_losses(steps, losses):
    plt.plot(steps, losses, label='')
    plt.xlabel('Iteration Steps')
    plt.ylabel('Average Loss')
    plt.title('Training CLIP Loss Curve')
    # plt.legend()
    plt.savefig(os.path.join('visuals/loss', f'{os.path.splitext(os.path.basename(args.loss_path))[0]}_loss.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    parser.add_argument('--loss_path', type=str, required=True)
    args = parser.parse_args()

    losses = []
    with open(os.path.join(args.loss_path), 'r') as f:
        for line in f:
            line = line.strip()
            index = line.find('train/loss_clip=')
            if index >= 0:
                try:
                    losses.append(float(line[index + 16:-1]))
                except ValueError:
                    pass

    plot_losses(np.arange(len(losses)), np.array(losses))
