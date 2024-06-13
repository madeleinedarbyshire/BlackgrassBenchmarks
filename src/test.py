import argparse
import math
import time
import torch
import torch.nn as nn

from model import full_precision_model
from torchvision import models
from utils import load_data

def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

def test(model, dataloader, device, channels, resolution, filename, model_name, batch_size=1):

    spectrum_name = ''.join([c[0]+c[-1] for c in channels])

    tp, tn, fn, fp = 0.0, 0.0, 0.0, 0.0
    running_time = 0.0
    total = 0.0
    for input_data, labels in dataloader:
        input_data = input_data.to(device)
        labels = labels.to(device)
        t1 = time.time()
        result = model(input_data)
        running_time += time.time() - t1
        _, preds = torch.max(result, 1)
        tp += torch.sum((preds == 1) & (labels.data == 1)).item()
        tn += torch.sum((preds == 0) & (labels.data == 0)).item()
        fn += torch.sum((preds == 0) & (labels.data == 1)).item()
        fp += torch.sum((preds == 1) & (labels.data == 0)).item()
        total += input_data.size(0)

    acc = (tp + tn) / (tp + fn + tn + fp)
    bg_precision = tp / (tp + fp)
    bg_recall = tp / (tp + fn)
    nbg_precision = tn / (tn + fn)
    nbg_recall = tn / (tp + fn)

    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


    inference_speed = 1 / (running_time / total)
    file = open(filename, 'a')
    file.write(f'Accuracy: {acc:.4f}\n')
    file.write(f'BG Pr: {bg_precision:.4f}\n')
    file.write(f'BG Re: {bg_recall:.4f}\n')
    file.write(f'No BG Pr: {nbg_precision:.4f}\n')
    file.write(f'Np BG Re: {nbg_recall:.4f}\n')
    file.write(f'MCC: {mcc:.4f}\n')
    file.write(f'-------------------------------------------\n')
    file.close()

def main(img_root, device, model_name, train_percentage, field, run_index=None):
    filename = f'results.txt'
    print('Writing results to file:', filename)
    file = open(filename, 'a')
    file.write(f'Model Name: {model_name}\n')
    file.write(f'Train Percentage: {train_percentage}\n')
    file.write(f'Run Index: {run_index}\n')
    file.write(f'Field: {field}\n')
    file.close()

    batch_size = 2
    resolution = 512
    channels = ['red', 'green', 'blue', 'nir', 'red_edge']

    spectrum_name = ''.join([c[0]+c[-1] for c in channels])
    if field:
        dataloaders = load_data(resolution, channels, ['test'], f'resources/test_field{field}.csv', path=img_root, batch_size=batch_size, test_device=False)
    else:
        dataloaders = load_data(resolution, channels, ['test'], 'resources/dataset_121023.csv', path=img_root, batch_size=batch_size, test_device=False)

    if run_index:
        model_path = f'models/121023_{train_percentage}/{model_name}_50_{resolution}_{spectrum_name}_{run_index}_best.pth'
    else:
        model_path = f'models/121023_{train_percentage}/{model_name}_50_{resolution}_{spectrum_name}_best.pth'
    model = full_precision_model(channels, model_name)
    model.load_state_dict(torch.load(model_path))

    model = model.to(device)
    model.eval()

    
    test(model, dataloaders['test'], device, channels, resolution, filename, model_name, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', nargs='+', type=str, default=['red', 'green', 'blue', 'nir', 'red_edge'])
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--img-root', type=str, default='../data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--precision', type=str, default='float32')
    parser.add_argument('--model-name', type=str, default='resnet50')
    parser.add_argument('--train-percentage', type=int, default=100)
    parser.add_argument('--field', type=str)
    parser.add_argument('--run-index', type=int)
    args = parser.parse_args()
    main(args.img_root, args.device, args.model_name, args.train_percentage, args.field, args.run_index)