import os
from pathlib import Path
from typing import Literal, Optional

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.v2 as T
import pandas as pd


class FashionMNIST(Dataset):
    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        root: Path = Path(os.path.dirname(__file__)) / ".." / "data",
        transform: Optional[T.Transform] = None,
    ):
        file_name = (
            root / "fashion-mnist_test.csv"
            if mode == "test"
            else root / "fashion-mnist_train.csv"
        )
        df = pd.read_csv(file_name)
        self.mode = mode
        self.transform = transform
        if mode in ["train", "val"]:
            if mode == "train":
                indices = df.sample(frac=0.8, random_state=42).index
            else:
                indices = df.drop(index=df.sample(frac=0.8, random_state=42).index).index

            self.x = torch.from_numpy(
                df.drop(columns=["label"]).iloc[indices].to_numpy().astype("int8")
            )
            self.y = torch.from_numpy(
                df.loc[indices, "label"].to_numpy().astype("int8")
            )
        else:
            self.x = torch.from_numpy(df.to_numpy().astype("int8"))

        assert torch.equal(
            torch.tensor(self.x.shape[1:]), torch.tensor([28 * 28])
        ), "Input shape must be 28x28 pixels"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].view(1, 28, 28).float() / 255.0
        if self.transform:
            x = self.transform(x)
        if self.mode in ["train", "val"]:
            y = self.y[idx].long()
            return x, y
        else:
            return x


train_transform = T.Compose(
    [
        T.ToImage(),
        T.Normalize(mean=[0.5], std=[0.5]),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomCrop(size=(28, 28), padding=4),
    ]
)

test_transform = T.Compose(
    [
        T.ToImage(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ]
)

labelNames = [
    "top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]
