import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

class JotformDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir = None, img_height = 100, img_width = 100):
        paths = []
        for r, d, f in os.walk(root_dir):
            for file in f:
                if file.endswith(".jpg"):
                    paths.append(os.path.join(r, file))

        self.paths = paths
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))

        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        return image

    def jotform_dataset_collate_fn(batch):
        images = zip(*batch)
        images = torch.stack(images, 0)
        return images
