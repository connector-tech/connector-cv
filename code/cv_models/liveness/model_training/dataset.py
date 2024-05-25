import cv2
import torch
from torch.utils.data import Dataset


class PersonDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.transform = transform
        self.image_paths = df["path"].tolist()

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)

        if self.transform:
            image = self.transform(image=image)["image"]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        label = self.df.iloc[index]["label"]
        return image, label

    def __len__(self) -> int:
        return len(self.df)
