from pathlib import Path
from typing import Callable, List, Optional, Tuple
import pickle
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def _unpickle(path: Path):
    """
    Unpickle the given file and return the data.
    :param path: Path to the file to unpickle.
    :return: Unpickled data.
    """
    with path.open("rb") as fh:
        return pickle.load(fh, encoding="bytes")

class Cifar100Dataset(Dataset):
    def __init__(self, root_dir: str = "dataset/cifar100", split: str = "train",
                 transform: Optional[Callable] = None):
        """
        Initialize the CIFAR-100 dataset.
        :param root_dir: Root directory containing CIFAR-100 files.
        :param split: Which split to load ("train" or "test").
        :param transform: Optional transform to apply to each image.
        """
        self.root = Path(root_dir)
        self.split = split
        self.transform = transform

        file_path = self.root / split
        if not file_path.exists():
            raise FileNotFoundError(f"Expected {file_path} to exist. Check dataset placement.")

        batch = _unpickle(file_path)
        data = batch[b"data"]  # (N, 3072) uint8
        self.images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # -> (N, 32, 32, 3)
        self.labels = batch[b"fine_labels"]

    def __len__(self) -> int:
        """
        Returns the total number of samples.
        :return: Length of the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Retrieve a single image and label by index.
        :param idx: Index of the sample to fetch.
        :return: Tuple (image, label) with transform applied if provided.
        """
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class Cifar100Loader:
    def __init__(self, root_dir: str = "dataset/cifar100", batch_size: int = 128,
                 num_workers: int = 2, pin_memory: bool = True, use_augmentation: bool = True):
        """
        Configure loaders and transforms for CIFAR-100.
        :param root_dir: Root directory for CIFAR-100 files.
        :param batch_size: Batch size for data loaders.
        :param num_workers: Number of worker processes for loading.
        :param pin_memory: Whether to pin memory in data loaders.
        :param use_augmentation: Whether to apply train-time augmentation.
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_augmentation = use_augmentation

        self.train_transform, self.test_transform = self._build_transforms()
        self.label_names = self._load_label_names()

    def _build_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Create train and test transforms with optional augmentation.
        :return: Tuple (train_transform, test_transform).
        """
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        train_aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ] if self.use_augmentation else []

        train_tf = transforms.Compose(train_aug + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return train_tf, test_tf

    def _load_label_names(self) -> List[str]:
        """
        Load fine label names from the CIFAR-100 meta file.
        :return: List of label names as strings.
        """
        meta_path = Path(self.root_dir) / "meta"
        if not meta_path.exists():
            raise FileNotFoundError(f"Expected meta file at {meta_path}.")
        meta = _unpickle(meta_path)
        return [name.decode("utf-8") for name in meta[b"fine_label_names"]]

    def get_datasets(self) -> Tuple[Cifar100Dataset, Cifar100Dataset]:
        """
        Build train and test Dataset instances.
        :return: Tuple (train_dataset, test_dataset).
        """
        train_ds = Cifar100Dataset(self.root_dir, split="train", transform=self.train_transform)
        test_ds = Cifar100Dataset(self.root_dir, split="test", transform=self.test_transform)
        return train_ds, test_ds

    def get_dataloaders(self, shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Build train and test DataLoader instances.
        :param shuffle_train: Whether to shuffle the training data.
        :return: Tuple (train_loader, test_loader).
        """
        train_ds, test_ds = self.get_datasets()
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return train_loader, test_loader

if __name__ == "__main__":
    loader = Cifar100Loader()
    train_loader, test_loader = loader.get_dataloaders()
    x, y = next(iter(train_loader))
    print(f"Train batch: images {x.shape}, labels {y.shape}")
    print(f"Number of classes: {len(loader.label_names)}")

# from src.Task2_PyTorch_DL.cifar100_loader import Cifar100Loader
# 
# loader = Cifar100Loader()
# train_dataset, test_dataset = loader.get_datasets()