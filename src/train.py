import models
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from engine import train, validate
from dataset_class import IntracranialDataset
from torch.utils.data import DataLoader

#
matplotlib.style.use('ggplot')

# Point to the relevent test label data and DICOM files
train_csv = pd.read_csv('../input/label_dataset/prototype_train_labels.csv')
data_path = '../input/images'

# initialize computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize model
model = models.model_fxn(pretrained=True, requires_grad = False).to(device)
# ResNext101 is 340 Mb

# Learning parameters
lr = 0.001
epochs = 2
batch_size = 8
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()


#train dataset
train_data = IntracranialDataset(
    train_csv, path=data_path, train=True, test=False
)

#Validation
valid_data = IntracranialDataset(
    train_csv, path=data_path, train=False, test=False
)

#train data loader
train_loader = DataLoader(
    train_data,
    batch_size = batch_size,
    shuffle = True
)

# validation data loader
valid_loader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle = False
)

# start training and validation
train_loss = []
valid_loss = []
for epoch in range(epochs):
  print(f"Epoch {epoch+1} of {epochs}")
  train_epoch_loss = train(model, train_loader, optimizer, criterion, train_data, device)
  valid_epoch_loss = validate(model, valid_loader, criterion, valid_data, device)
  train_loss.append(train_epoch_loss)
  valid_loss.append(valid_epoch_loss)
  print(f'Train Loss: {train_epoch_loss:.4f}')
  print(f'Val Loss: {valid_epoch_loss:.4f}')

# Save trained model to disk
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
}, '../output/model.pth')

# Plot and save train/validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'../output/loss_lr{lr}_e{epochs}_b{batch_size}.png')