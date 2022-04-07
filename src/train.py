
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
import os
#
matplotlib.style.use('ggplot')


# Point to the relevent test label data and DICOM files
#train_csv = pd.read_csv('../input/label_dataset/prototype_train_labels.csv')
data_path = '../all_sites/'

# initialize computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize model
model = models.model_fxn(pretrained=True, requires_grad = False).to(device)
# ResNext101 is 340 Mb

#train dataset
train_data = IntracranialDataset(
    path=data_path, train=True, test=False
)

#Validation
valid_data = IntracranialDataset(
    path=data_path, train=False, test=False
)

## Learning parameters
lr = 0.0003
epochs = 5


batch_size = 16
optimizer = optim.Adam(model.parameters(), lr=lr)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

## Define loss criterion with weights
loss_weights=train_data.loss_weights.to(device)
print(f'loss weights: {loss_weights}')
criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
#criterion = nn.BCEWithLogitsLoss()

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

# Plot and save train/validation line graphs
def plot_loss(training_loss, validation_loss, training_acc, validation_acc):
    fig, axs = plt.subplots(1,2,figsize=(10,7))
    axs[0].plot(training_loss, color='orange', label='train loss')
    axs[0].plot(validation_loss, color='red', label='validataion loss')
    axs[1].plot(training_acc, color = 'orange', label='train accuracy')
    axs[1].plot(validation_acc, color = 'red', label='valid accuracy')
    axs[0].set_title("Loss per Epoch")
    axs[1].set_title("Accuracy per Epoch")
    axs[0].legend(loc='center left')
    axs[1].legend(loc='center left')
    fig.savefig(f'../output/{runtime_day}_{epochs}_b{batch_size}.png')

if os.path.exists('../output'):
    print('output path exists')
else:
    print("\noutput directory does not exist")
    os.mkdir('../output/')
    print("Directory ../output/ created\n")

## Train
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_results = train(model, train_loader, optimizer, criterion, train_data, device)
    valid_results = validate(model, valid_loader, criterion, valid_data, device)
    #Save values for output
    train_loss.append(train_results['train_loss'])
    valid_loss.append(valid_results['val_loss'])

    ## Calculate accuracy (correct label for any of the 6 subclasses)
    flat_train_pred = np.array([item for sublist in train_results['pred'] for item in sublist])
    flat_train_label = np.array([item for sublist in train_results['label'] for item in sublist])
    train_acc = metrics.f1_score(flat_train_label, np.where(flat_train_pred > 0.5, 1, 0))
    
    flat_val_pred = np.array([item for sublist in valid_results['pred'] for item in sublist])
    flat_val_label = np.array([item for sublist in valid_results['label'] for item in sublist])
    val_acc = metrics.f1_score(flat_val_label, np.where(flat_val_pred > 0.5, 1, 0))

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
    plot_loss(train_loss, valid_loss, train_acc, val_acc)
    epoch_metrics = pd.DataFrame(data = {'epoch':list(range(1,epochs+1)),'learning_rate':learning_rate, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'valid_loss':valid_loss, 'valid_accuracy':valid_accuracy})
    epoch_metrics.to_csv(f'../output/{runtime_day}_epoch_metrics.csv', index = False)
    # Save trained model to disk
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f'../output/{epoch}_model.pt')
    torch.cuda.empty_cache()

#fin