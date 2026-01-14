import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gdown

# Google Drive file ID of stl10_binary.tar.gz
# Example link:
# https://drive.google.com/file/d/FILE_ID/view
GDRIVE_FILE_ID = "https://drive.google.com/file/d/1qrEgd5-O-xE04PJ9uGbADK3NFAa0LGA7/view?usp=sharing"

DATA_ROOT = "./stl10_data"
ARCHIVE_PATH = os.path.join(DATA_ROOT, "stl10_binary.tar.gz")
EXTRACT_PATH = os.path.join(DATA_ROOT, "stl10_binary")

BATCH_SIZE = 128
NUM_WORKERS = 4

def download_and_extract_stl10():
    os.makedirs(DATA_ROOT, exist_ok=True)

    if not os.path.exists(ARCHIVE_PATH):
        print("Downloading STL-10 from Google Drive...")
        gdown.download(
            id='1qrEgd5-O-xE04PJ9uGbADK3NFAa0LGA7',
            output=ARCHIVE_PATH,
            quiet=False
        )
    else:
        print("Archive already exists. Skipping download.")

    if not os.path.exists(EXTRACT_PATH):
        print("Extracting STL-10 archive...")
        import tarfile
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(DATA_ROOT)
    else:
        print("STL-10 already extracted.")


class STL10BinaryDataset(Dataset):
    def __init__(self, data_path, label_path=None):
        with open(data_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint8)

        self.images = data.reshape(-1, 3, 96, 96).astype(np.float32) / 255.0

        if label_path is not None:
            with open(label_path, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8)
            self.labels = labels - 1
        else:
            self.labels = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.images[idx])
        if self.labels is None:
            return x
        return x, int(self.labels[idx])

def get_stl10_dataloaders():
    train_X = os.path.join(EXTRACT_PATH, "train_X.bin")
    train_y = os.path.join(EXTRACT_PATH, "train_y.bin")
    test_X  = os.path.join(EXTRACT_PATH, "test_X.bin")
    test_y  = os.path.join(EXTRACT_PATH, "test_y.bin")
    unlab_X = os.path.join(EXTRACT_PATH, "unlabeled_X.bin")

    train_dataset = STL10BinaryDataset(train_X, train_y)
    test_dataset = STL10BinaryDataset(test_X, test_y)
    unlabeled_dataset = STL10BinaryDataset(unlab_X)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, unlabeled_loader


if __name__ == "__main__":
    download_and_extract_stl10()

    train_loader, test_loader, unlabeled_loader = get_stl10_dataloaders()

    images, labels = next(iter(train_loader))
    print("Train batch:", images.shape, labels.shape)

    images = next(iter(unlabeled_loader))
    print("Unlabeled batch:", images.shape)

# expected output:
# pls do check if you get this output before you proceed with the model training

# Downloading STL-10 from Google Drive...
# Extracting STL-10 archive...
# Train batch: torch.Size([128, 3, 96, 96]) torch.Size([128])
# Unlabeled batch: torch.Size([128, 3, 96, 96])
