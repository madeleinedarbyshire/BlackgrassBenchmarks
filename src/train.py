from __future__ import print_function, division

import argparse
import copy
import math
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
import torch.backends.cudnn as cudnn

from model import full_precision_model
from utils import load_data

def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, model_filename, num_epochs=25, save=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        epoch_start = time.time() 

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Learning Rate: {scheduler.get_last_lr()[0]:.4f} Total: {total:.0f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print(f' Best Acc: {best_acc:.4f}')
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_filename + '_best.pth')

        epoch_duration = time.time() - epoch_start
        print(f'Epoch complete in {epoch_duration // 60:.0f}m {epoch_duration % 60:.0f}s')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model_wts = copy.deepcopy(model.state_dict())
    # torch.save(model_wts, model_filename + '_last.pth')
    model.load_state_dict(model_wts)
    return model

def setup_and_train_model(batch_size, channels, img_root, num_epochs, resolution, max_lr, min_lr, momentum, model_name, train_percentage, run_index):

    # Set random seeds for CPU
    torch.manual_seed(run_index)
    np.random.seed(run_index)

    # Check if GPU is available and set seed
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(run_index)

    spectrum_name = ''.join([c[0]+c[-1] for c in channels])
    train_percentage_name = str(math.floor(train_percentage))
    os.makedirs(f'out/121023_{train_percentage_name}', exist_ok=True)
    txt_filename = f'out/121023_{train_percentage_name}/{model_name}_{num_epochs}_{resolution}_{spectrum_name}_{run_index}.log'
    model_filename = f'models/121023_{train_percentage_name}/{model_name}_{num_epochs}_{resolution}_{spectrum_name}_{run_index}'
    with open(txt_filename, 'w') as sys.stdout:
        dataloaders = load_data(resolution, channels, ['train', 'val'], f'resources/train_{train_percentage_name}_121023.csv', path=img_root, class_path='resources/labels.txt', batch_size=batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = full_precision_model(channels,model_name)
        model = model.to(device)
        optimizer_ft = SGD(model.parameters(), lr=max_lr, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        T_max = ((11752 * (train_percentage / 100)) / batch_size) * num_epochs
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max, eta_min=min_lr, last_epoch=-1, verbose=False)
        train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, model_filename, num_epochs=num_epochs)

def main(batch_size, channels, img_root, num_epochs, resolution, max_lr, min_lr, momentum, model_name, train_powerset, train_percentage, run_index):
    if train_powerset:
        for channels in powerset(['red', 'green', 'blue', 'nir', 'red_edge']):
            print(f'Channels: {channels}')
            setup_and_train_model(batch_size, channels, img_root, num_epochs, resolution, max_lr, min_lr, momentum, model_name)
            sys.stdout = sys.__stdout__
    else:
        print(f'Training model... {channels}')
        print(f'Channels: {channels}')
        print(f'Epochs: {num_epochs}')
        print(f'Resolution: {resolution}')
        print(f'Batch Size: {batch_size}')
        print(f'Weights: {model_name}')
        setup_and_train_model(batch_size, channels, img_root, num_epochs, resolution, max_lr, min_lr, momentum, model_name, train_percentage, run_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--channels', nargs="+", type=str, default=['red', 'green', 'blue', 'nir', 'red_edge'])
    parser.add_argument('--img-root', type=str, default='../data')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--max-lr', type=float, default=0.001)
    parser.add_argument('--min-lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--model-name', type=str, default='resnet18')
    parser.add_argument('--train-powerset', type=bool)
    parser.add_argument('--train-percentage', type=float)
    parser.add_argument('--run-index', type=int, default=1)
    args = parser.parse_args()
    main(args.batch_size, args.channels, args.img_root, args.num_epochs, args.resolution, args.max_lr, args.min_lr, args.momentum, args.model_name, args.train_powerset, args.train_percentage, args.run_index)