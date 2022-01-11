import models
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from engine import train, validate
from dataset_class import IntracranialDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from datetime import date
from sklearn import metrics
runtime_day = date.today().strftime("%b-%d-%Y")
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
lr = 0.01
epochs = 12
batch_size = 16
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

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
train_accuracy = []
valid_accuracy = []
learning_rate = []

for epoch in range(epochs):
  print(f"Epoch {epoch+1} of {epochs}")
  train_results = train(model, train_loader, optimizer, criterion, scheduler, train_data, device)
  valid_results = validate(model, valid_loader, criterion, valid_data, device)
  #Save values for output
  train_loss.append(train_results['train_loss'])
  valid_loss.append(valid_results['val_loss'])

  ## Calculate accuracy (correct label for any of the 6 subclasses)
  flat_train_pred = np.array([item for sublist in train_results['pred'] for item in sublist])
  flat_train_label = np.array([item for sublist in train_results['label'] for item in sublist])
  train_acc = metrics.accuracy_score(flat_train_label, np.where(flat_train_pred > 0.5, 1, 0))
  flat_val_pred = np.array([item for sublist in valid_results['pred'] for item in sublist])
  flat_val_label = np.array([item for sublist in valid_results['label'] for item in sublist])
  val_acc = metrics.accuracy_score(flat_val_label, np.where(flat_val_pred > 0.5, 1, 0))

  train_accuracy.append(train_acc)
  valid_accuracy.append(val_acc)
  # Print epoch results
  epoch_lr = optimizer.param_groups[0]["lr"]
  learning_rate.append(epoch_lr)
  print(f'\nEpoch {epoch+1} learning rate: {epoch_lr}')
  print(f'Train Loss: {train_results["train_loss"]:.4f}')
  print(f'Val Loss: {valid_results["val_loss"]:.4f}')
  print(f'Training accuracy = {train_acc:.5f}')
  print(f'Validation accuracy = {val_acc:.5f}')
  print('------------------\n')
  #
  torch.cuda.empty_cache()

##
epoch_metrics = pd.DataFrame(data = {'epoch':list(range(1,epochs+1)),'learning_rate':learning_rate, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'valid_loss':valid_loss, 'valid_accuracy':valid_accuracy})
epoch_metrics.to_csv(f'../output/{runtime_day}_epoch_metrics.csv', index = False)

# Save trained model to disk
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
}, '../output/model.pth')

# Plot and save train/validation line graphs
fig, axs = plt.subplots(1,2,figsize=(10,7))
axs[0].plot(train_loss, color='orange', label='train loss')
axs[0].plot(valid_loss, color='red', label='validataion loss')
axs[1].plot(train_accuracy, color = 'orange', label='train accuracy')
axs[1].plot(valid_accuracy, color = 'red', label='valid accuracy')
axs[0].set_title("Loss per Epoch")
axs[1].set_title("Accuracy per Epoch")
axs[0].legend(loc='center left')
axs[1].legend(loc='center left')
fig.savefig(f'../output/{runtime_day}_{epochs}_b{batch_size}.png')