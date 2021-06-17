import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

class OCR_Dataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__ (self, mode = None, root_dir = None, img_height = 100, img_width = 100):

        mapping = {}

        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(tqdm(fr.readlines())):
                mapping[i] = line.strip()

        if mode == 'train':
            path = 'annotation_train.txt'
        elif mode == 'val':
            path = 'annotation_val.txt'
        elif mode == 'test':
            path = 'annotation_test.txt'
        else:
            raise Exception("Incorrect argument for variable mode!")

        paths = []
        texts = []

        with open(os.path.join(root_dir, path), 'r') as fr:
            for line in tqdm(fr.readlines()[:1000]):
                line_stripped = line.strip()

                cur_path, index = line_stripped.split(' ')

                cur_path = os.path.join(root_dir, cur_path[2:])
                index = int(index)

                paths.append(cur_path)
                texts.append(mapping[index])

        self.paths = paths
        self.texts = texts
        self.mode = mode
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

        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)

            return image, target, target_length
        else:
            return image

def ocr_dataset_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)

    return images, targets, target_lengths
