from __future__ import print_function, division

import cv2
import numpy as np
import os
import pandas as pd
import torch
from torchvision import datasets
from torchvision import transforms
from typing import Any, Callable, Dict, List, Optional, Tuple
from typing import Union

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def multispectral_image_loader(img_paths):
    img = []
    try:
        for i, path in enumerate(img_paths):
            band = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
            img.append(band)
    except:
        print('Could not load:', path)
        return

    return torch.from_numpy(np.array(img))

class PandasDataset(datasets.ImageFolder):

    def __init__(
        self,
        root: str,
        channels: List[str],
        class_path: str,
        metadata: pd.DataFrame,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = multispectral_image_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        test_device: bool = True
    ):
        self.metadata = metadata
        self.channels = channels
        self.set = set
        self.class_path = class_path
        self.test_device = test_device
        super().__init__(
            root,
            loader=loader,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = []
        class_to_idx = {}
        with open(self.class_path) as f:
            class_dict = eval(f.read())
            classes = [class_dict[i] for i in range(len(class_dict))]
            class_to_idx = {class_name : i for i, class_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            class_metadata = self.metadata[self.metadata['binary_class'] == target_class].copy()
            class_metadata.sort_values(by=['name'], inplace=True)
            for _, row in class_metadata.iterrows():
                if self.test_device == True:
                    filename_prefix = os.path.join(directory, row['binary_class'], row['field'] + '_' + row['date'] + '_' + row['class'] + '_')
                    paths = [filename_prefix + row[band] for band in self.channels]
                else:
                    target_directory = os.path.join(directory, row['field'], row['date'], row['class'])
                    paths = [os.path.join(target_directory, i) for i in [row[x] for x in self.channels]]            
                item = paths, class_index
                instances.append(item)

        return instances

def generate_data_transforms(resolution, channels):
    means = {'red': 0.2032, 'green': 0.2502, 'blue': 0.2250, 'nir' :0.3375, 'red_edge' : 0.2863}
    stds = {'red': 0.1095, 'green': 0.1281, 'blue': 0.1133, 'nir' : 0.1768, 'red_edge' : 0.1414}
    center_crop = transforms.CenterCrop(512)
    normalize = transforms.Normalize([means[b] for b in channels], [stds[b] for b in channels])
    if resolution == 512:
        train = transforms.Compose([center_crop, transforms.RandomHorizontalFlip(), normalize])
        val = transforms.Compose([center_crop, normalize])
    else:
        train = transforms.Compose([center_crop, transforms.Resize(resolution), transforms.RandomHorizontalFlip(), normalize])
        val = transforms.Compose([center_crop, transforms.Resize(resolution), normalize])
    return {'train' : train, 'val' : val, 'test': val, 'cal':val}

def load_data(resolution, channels, dataset_types, metadata_file, path='../data', class_path='resources/labels.txt', batch_size=64, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=4, test_device=False):
    data_dir = path
    metadata = pd.read_csv(metadata_file)
    print('Loading datasets...')
    if resolution:
        data_transforms = generate_data_transforms(resolution, channels)
        image_datasets = {x: PandasDataset(root=data_dir, metadata=metadata[metadata['dataset_type'] == x], transform=data_transforms[x], channels=channels, class_path=class_path, test_device=test_device) for x in dataset_types}
    else:
        image_datasets = {x: PandasDataset(root=data_dir, metadata=metadata[metadata['dataset_type'] == x], transform=None, channels=channels, class_path=class_path, test_device=test_device) for x in dataset_types}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, prefetch_factor=prefetch_factor) for x in dataset_types}
    dataset_sizes = {x: len(image_datasets[x]) for x in dataset_types}
    class_names = image_datasets[dataset_types[0]].classes
    print('Dataset sizes: ', dataset_sizes)
    print('Class names: ', class_names)
    return dataloaders