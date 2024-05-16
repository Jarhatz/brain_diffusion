import os, argparse
import pandas as pd
import torch
import transformers
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000


# python code/eval_generations.py --results_path results/eval/cos_300_5e-5


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    parser.add_argument('--results_path', type=str, required=True)
    args = parser.parse_args()

    feature_extractor = transformers.ViTImageProcessor.from_pretrained('google/vit-large-patch32-384', cache_dir = 'cache')
    model = transformers.ViTForImageClassification.from_pretrained('google/vit-large-patch32-384', cache_dir = 'cache').to(device)

    top_1_acc = 0
    top_3_acc = 0
    gen_count = 0
    gt_class_id = None
    
    for filename in sorted(os.listdir(args.results_path)):
        if filename.startswith('test'):
            image = Image.open(os.path.join(args.results_path, filename))
            inputs = feature_extractor(images=image, return_tensors="pt").to(device)
            logits = model(**inputs).logits[0,:]
            if filename.endswith('-0.png'):
                gt_class_id = logits.argmax(-1).item()
                # print(f'filename: {filename}\tpredicted id: {gt_class_id}\tpredicted class: {model.config.id2label[gt_class_id]}')
            else:
                gen_count += 1
                _, top_indeces = torch.topk(logits, k=3)
                # print(f'filename: {filename}\ttop 3: {top_indeces}')
                if top_indeces[0] == gt_class_id:
                    top_1_acc += 1
                    top_3_acc += 1
                elif torch.any(top_indeces == gt_class_id):
                    top_3_acc += 1
    print(f'Top 1 Acc: {top_1_acc / gen_count}\tTop 3 Acc: {top_3_acc / gen_count}')
        