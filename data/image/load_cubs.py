# !pip install -q gdown

import os
import gdown
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


# Google Drive FOLDER ID (not file ID!)
# Example folder link:
# https://drive.google.com/drive/folders/FOLDER_ID
GDRIVE_FOLDER_ID = "PUT_YOUR_FOLDER_ID_HERE"

DATA_ROOT = "./data"
CUB_ROOT = os.path.join(DATA_ROOT, "CUB_200_2011")

def download_cub_from_gdrive_folder():
    os.makedirs(DATA_ROOT, exist_ok=True)

    if os.path.exists(CUB_ROOT):
        print("CUB_200_2011 already exists. Skipping download.")
        return

    print("Downloading CUB_200_2011 folder from Google Drive...")

    gdown.download_folder(
        id=GDRIVE_FOLDER_ID,
        output=DATA_ROOT,
        quiet=False,
        use_cookies=False
    )

    print("Download completed.")
    print("CUB dataset available at:", CUB_ROOT)

def verify_cub_structure(root):
    required_files = [
        "images.txt",
        "image_class_labels.txt",
        "train_test_split.txt"
    ]

    required_dirs = [
        "images"
    ]

    for f in required_files:
        path = os.path.join(root, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    for d in required_dirs:
        path = os.path.join(root, d)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Missing directory: {path}")

    print("CUB directory structure verified.")

if __name__ == "__main__":
    download_cub_from_gdrive_folder()
    verify_cub_structure(CUB_ROOT)

    # Use this ROOT in your existing code
    ROOT = CUB_ROOT
    print("Use ROOT =", ROOT)

ROOT = "./data/CUB_200_2011"

train_data = CUB(ROOT, 'train', SPLIT_RATIO, RANDOM_SEED, transform=trans_train)
valid_data = CUB(ROOT, 'valid', SPLIT_RATIO, RANDOM_SEED, transform=trans_test)

class CUB(Dataset):
    def __init__(
        self,
        root,
        dataset_type="train",   # "train" | "valid" | "test"
        train_ratio=1.0,
        valid_seed=123,
        transform=None,
        target_transform=None
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # Load metadata
        df_img = pd.read_csv(
            os.path.join(root, "images.txt"),
            sep=" ", header=None, names=["ID", "Image"], index_col=0
        )
        df_label = pd.read_csv(
            os.path.join(root, "image_class_labels.txt"),
            sep=" ", header=None, names=["ID", "Label"], index_col=0
        )
        df_split = pd.read_csv(
            os.path.join(root, "train_test_split.txt"),
            sep=" ", header=None, names=["ID", "Train"], index_col=0
        )

        df = pd.concat([df_img, df_label, df_split], axis=1)
        df["Label"] = df["Label"] - 1  # 1–200 → 0–199

        # Split logic
        if dataset_type == "test":
            df = df[df["Train"] == 0]

        elif dataset_type in ["train", "valid"]:
            df = df[df["Train"] == 1]

            if train_ratio < 1.0:
                np.random.seed(valid_seed)
                indices = np.random.permutation(len(df))
                split_idx = int(len(df) * train_ratio)

                if dataset_type == "train":
                    df = df.iloc[indices[:split_idx]]
                else:
                    df = df.iloc[indices[split_idx:]]
            elif dataset_type == "valid":
                raise ValueError("train_ratio must be < 1 for validation split")

        else:
            raise ValueError("dataset_type must be train | valid | test")

        self.img_names = df["Image"].tolist()
        self.labels = df["Label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        target = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

trans_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

trans_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

ROOT = "./data/CUB_200_2011"   # path to downloaded folder
BATCH_SIZE = 64
NUM_WORKERS = 4
SPLIT_RATIO = 0.9
RANDOM_SEED = 123

train_dataset = CUB(
    ROOT,
    dataset_type="train",
    train_ratio=SPLIT_RATIO,
    valid_seed=RANDOM_SEED,
    transform=trans_train
)

valid_dataset = CUB(
    ROOT,
    dataset_type="valid",
    train_ratio=SPLIT_RATIO,
    valid_seed=RANDOM_SEED,
    transform=trans_test
)

test_dataset = CUB(
    ROOT,
    dataset_type="test",
    transform=trans_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=NUM_WORKERS
)
images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

# sanity check: pls ensure that you get this output below:
# torch.Size([64, 3, 224, 224]) torch.Size([64])
