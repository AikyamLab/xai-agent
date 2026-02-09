import numpy as np 
import pandas as pd 
import torch, os
import numpy as np
from torchvision import transforms
from torch.utils import data
from PIL import Image
import pandas as pd
import torch.nn.functional as F
from torch import optim

import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 64
NUM_WORKERS = 4
SPLIT_RATIO = 0.9
RANDOM_SEED = 123
CLASS_NUM = 200
ROOT = r'../input/cub2002011/CUB_200_2011' # please replace it with your working directory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CUB():
    def __init__(self, root, dataset_type='train', train_ratio=1, valid_seed=123, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        df_img = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ', header=None, names=['ID', 'Image'], index_col=0)
        df_label = pd.read_csv(os.path.join(root, 'image_class_labels.txt'), sep=' ', header=None, names=['ID', 'Label'], index_col=0)
        df_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', header=None, names=['ID', 'Train'], index_col=0)
        df = pd.concat([df_img, df_label, df_split], axis=1)
        # relabel
        df['Label'] = df['Label'] - 1

        # split data
        if dataset_type == 'test':
            df = df[df['Train'] == 0]
        elif dataset_type == 'train' or dataset_type == 'valid':
            df = df[df['Train'] == 1]
            # random split train, valid
            if train_ratio != 1:
                np.random.seed(valid_seed)
                indices = list(range(len(df)))
                np.random.shuffle(indices)
                split_idx = int(len(indices) * train_ratio) + 1
            elif dataset_type == 'valid':
                raise ValueError('train_ratio should be less than 1!')
            if dataset_type == 'train':
                df = df.iloc[indices[:split_idx]]
            else:       # dataset_type == 'valid'
                df = df.iloc[indices[split_idx:]]
        else:
            raise ValueError('Unsupported dataset_type!')
        self.img_name_list = df['Image'].tolist()
        self.label_list = df['Label'].tolist()
        # Convert greyscale images to RGB mode
        self._convert2rgb()

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.img_name_list[idx])
        image = Image.open(img_path)
        target = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def _convert2rgb(self):
        for i, img_name in enumerate(self.img_name_list):
            img_path = os.path.join(self.root, 'images', img_name)
            image = Image.open(img_path)
            color_mode = image.mode
            if color_mode != 'RGB':
                # image = image.convert('RGB')
                # image.save(img_path.replace('.jpg', '_rgb.jpg'))
                self.img_name_list[i] = img_name.replace('.jpg', '_rgb.jpg')

trans_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.7, 1), ratio=(3/4, 4/3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

trans_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# create dataset
train_data = CUB(ROOT, 'train', SPLIT_RATIO, RANDOM_SEED, transform=trans_train)
valid_data = CUB(ROOT, 'valid', SPLIT_RATIO, RANDOM_SEED, transform=trans_test)

print("Train: {}".format(len(train_data)))
print("Valid: {}".format(len(valid_data)))

'''
Uncomment this section, if needed for testing the above dataloader

# test - sanity check
from matplotlib import pyplot as plt

def imshow(image, label, ax=None, normalize=True):
    """show single along with label on an ax"""
    
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_title(label)

    return ax


def show_samples(images, labels, nrows=2, ncols=3, title=None, normalize=True):
    """ show multiple samples

    args:
        nrows (int, optional): number of row
        ncols (int, optional): number of column
        title (str, optional): title.
        normalize (bool, optional): whether the images are normalized
    """
    fig, axes = plt.subplots(nrows, ncols, facecolor='#ffffff', dpi=100)

    # .flat: to map samples to multi-dimensional axes
    for (ax, image, label) in zip(axes.flat, images, labels):
        ax = imshow(image, label, ax, normalize)

    fig.suptitle(title)
    fig.tight_layout = True
    fig.subplots_adjust(top=0.85, hspace=0.3)
    plt.show()

# show samples
train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
images, labels = next(iter(train_loader))
show_samples(images[0:6], labels[0:6], 2, 3, 'SHOW SOME SAMPLES')

'''

# create dataloader
train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
valid_loader = data.DataLoader(valid_data, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS)

test_data = CUB(ROOT, 'test', SPLIT_RATIO, RANDOM_SEED, transform=trans_test)
test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=NUM_WORKERS)

label_map = {i: f"class_{i}" for i in range(CLASS_NUM)}

import os

def load_cub_label_map(root):
    """
    Returns:
        dict: {0: 'Black_footed_Albatross', ..., 199: 'Common_Yellowthroat'}
    """
    label_map = {}
    classes_path = os.path.join(root, "classes.txt")

    with open(classes_path, "r") as f:
        for line in f:
            idx, name = line.strip().split(" ", 1)
            label_map[int(idx) - 1] = name  # convert to 0-based

    return label_map

label_map = load_cub_label_map(ROOT)

print(label_map)
