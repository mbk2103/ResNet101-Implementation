from custom_dataset import CustomDataset
from evaluate_visualization import EvaluateVisualization
from resnet_model import ResNet101v2, ResidualBlock
from resnet_trainer import ResNetTrainer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
  # Define transformations for data augmentation
  transform = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
  
  # Create custom dataset and DataLoader
  data_dir = "path/to/your/dataset"
  custom_dataset = CustomDataset(data_dir, transform=transform)
  train_size = int(0.8 * len(custom_dataset))
  val_size = len(custom_dataset) - train_size
  train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

  train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
  val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

  # Create test dataset and DataLoader
  test_dataset = CustomDataset("path/to/test/dataset", transform=transform)
  test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

  # Create ResNet101v2 model
  num_classes = len(custom_dataset.classes)
  resnet101v2_model = ResNet101v2(num_classes)

  # Define loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(resnet101v2_model.parameters(), lr=0.001)

  # Training and evaluation
  trainer = ResNetTrainer(
      model=resnet101v2_model,
      train_dataloader=train_dataloader,
      val_dataloader=val_dataloader,
      test_dataloader=test_dataloader,
      criterion=criterion,
      optimizer=optimizer,
      device='cuda' if torch.cuda.is_avaliable() else 'cpu'
  )

  num_epochs = 10
  trainer.train(num_epochs)
  trainer.evaluate()