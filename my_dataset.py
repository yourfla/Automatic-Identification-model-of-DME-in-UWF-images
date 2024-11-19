import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class AugmentedDiabeticMacularEdemaDataset(Dataset):
    """自定义数据集，包含数据增强"""

    def __init__(self, csv_file, root_dir, transform=None, num_augments=10, incorrect_samples=None, repeat_factor=1):
        self.df = pd.read_excel(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.num_augments = num_augments
        self.incorrect_samples = incorrect_samples
        self.repeat_factor = repeat_factor

        if incorrect_samples is not None:
            samples_to_append = []
            for sample in incorrect_samples:
                for _ in range(repeat_factor):
                    samples_to_append.append(self.df[self.df['image'] == sample])
            self.df = pd.concat([self.df] + samples_to_append, ignore_index=True)

    def __len__(self):
        return len(self.df) * self.num_augments

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx // self.num_augments, 0])
        image = Image.open(img_name)

        if image.mode != 'RGB':
            raise ValueError(f"image: {img_name} 不是 RGB 模式.")

        label = self.df.iloc[idx // self.num_augments, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
