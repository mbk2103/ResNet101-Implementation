import torch
from torchvision import datasets, transforms

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.transform = transform
    self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]