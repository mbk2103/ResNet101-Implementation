import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from evaluate_visualization import EvaluateVisualization

# Training class
class ResNetTrainer:
  def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, device='cuda'):
    self.model = model
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader
    self.test_dataloader = test_dataloader
    self.criterion = criterion
    self.optimizer = optimizer
    self.device = device
    self.evaluator = EvaluateVisualization()

  def train(self, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
      # Training code...
      self.model.train()
      for inputs, labels in self.train_dataloader:
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
      
      # Validation Code..
      self.model.eval()
      val_loss = 0.0
      val_samples = 0
      y_true = []
      y_pred = []

      with torch.no_grad():
        for inputs, labels in self.val_dataloader:
          inputs, labels = inputs.to(self.device), labels.to(self.device)

          outputs = self.model(inputs)
          loss = self.criterion(outputs, labels)
          val_loss += loss.item() * inputs.size(0)
          val_samples += inputs.size(0)

          _, predicted = torch.max(outputs, 1)
          y_true.extend(labels.cpu().numpy())
          y_pred.extend(predicted.cpu().numpy())

        val_losses.append(val_loss / val_samples)
    
    # Plotting loss curve
    self.evaluator.plot_loss_curve(train_losses, val_losses)

  def evaluate(self):
    self.model.eval()
    y_true_test = []
    y_pred_test = []

    with torch.no_grad():
      for inputs, labels in self.test_dataloader:
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        outputs = self.model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(predicted.cpu().numpy())

    # Plotting confusion matrix
    class_names = self.test_dataloader.dataset.classes
    self.evaluator.plot_confusion_matrix(y_true_test, y_pred_test, class_names)
    print('Evaluation finished.')