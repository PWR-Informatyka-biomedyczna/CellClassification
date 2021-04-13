import glob
import os


import torch
from torch.utils.data import Dataset
from PIL import Image


from core.dataset.transforms import default_transforms


class Data(Dataset):

    def __init__(self, path, classes, kind='train', target_size=(128, 128), transforms=default_transforms):
        self._class_map = self.__setup_classes(classes)
        self._data = self.__setup_data(path, kind)
        self._transforms = transforms(target_size)

    def __setup_data(self, path, kind):
        data = []
        for c in self._class_map.keys():
            paths = [(input_path, c) for input_path in glob.glob(os.path.join(path, c, kind, '*'))]
            data.extend(paths)
        return data

    def __setup_classes(self, classes):
        class_map = dict()
        num_classes = len(classes)
        for i, c in enumerate(classes):
            class_map[c] = torch.tensor(i)
        return class_map

    def __len__(self):
        return len(self._data)

    def __read_image(self, path):
        img = Image.open(path)
        return self._transforms(img)

    def __getitem__(self, item):
        path, label = self._data[item]
        img = self.__read_image(path)
        return img, self._class_map[label].long()

