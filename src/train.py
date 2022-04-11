
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
from datetime import datetime
from sklearn import metrics
import os
import sys
#
matplotlib.style.use('ggplot')
now = datetime.now()
# Point to the relevent test label data and DICOM files
#train_csv = pd.read_csv('../input/label_dataset/prototype_train_labels.csv')
data_path = '../all_sites/'
if os.path.exists('../output'):
    print('output path exists')
else:
    print("\noutput directory does not exist")
    os.mkdir('../output/')
    print("Directory ../output/ created\n")
sys.stdout = open('../output/output_log.txt', 'w')


# initialize computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize model
model = models.model_fxn(pretrained=True, requires_grad = False).to(device)
model = torch.nn.DataParallel(model).to(device)
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
epochs = 3


batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=lr)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

## Define loss criterion with weights
loss_weights=train_data.loss_weights.to(device)
print(f'\nBCE Loss weights: {loss_weights}')
criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)


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
running_train_roc = []
running_val_roc = []
running_train_prc = []
running_val_prc = []

# Plot and save train/validation line graphs
def plot_loss_f1(training_loss, validation_loss, training_acc, validation_acc):
    fig, axs = plt.subplots(1,2,figsize=(10,7))
    axs[0].plot(training_loss, color='orange', label='train loss')
    axs[0].plot(validation_loss, color='red', label='validataion loss')
    axs[1].plot(training_acc, color = 'orange', label='train accuracy')
    axs[1].plot(validation_acc, color = 'red', label='valid accuracy')
    axs[0].set_title("Loss per Epoch")
    axs[1].set_title("F1 Score per Epoch")
    axs[0].legend(loc='center left')
    axs[1].legend(loc='center left')
    fig.savefig(f'../output/{now.strftime("%b-%d-%Y-%H-%M")}_{epochs}_b{batch_size}_loss_F1.png')

def plot_roc_prc(training_roc, validation_roc, training_prc, validation_prc):
    fig, axs = plt.subplots(1,2,figsize=(10,7))
    axs[0].plot(training_roc, color='orange', label='train ROC auc')
    axs[0].plot(validation_roc, color='red', label='validataion ROC auc')
    axs[1].plot(training_prc, color = 'orange', label='train PRC auc')
    axs[1].plot(validation_prc, color = 'red', label='valid PRC auc')
    axs[0].set_title("AUROC per Epoch")
    axs[1].set_title("AUPRC per Epoch")
    axs[0].legend(loc='center left')
    axs[1].legend(loc='center left')
    fig.savefig(f'../output/{now.strftime("%b-%d-%Y-%H-%M")}_{epochs}_b{batch_size}_ROCPRC.png')

## Train
for epoch in range(epochs):
    print(f"\n{datetime.now()}")
    print(f"\nEpoch {epoch+1} of {epochs}")
    train_results = train(model, train_loader, optimizer, criterion, train_data, device)
    valid_results = validate(model, valid_loader, criterion, valid_data, device)
 

    #Save values for output
    train_loss.append(train_results['train_loss'])
    valid_loss.append(valid_results['val_loss'])

    ## Calculate accuracy (correct label for any of the 6 subclasses)
    flat_train_pred = np.array([item for sublist in train_results['pred'] for item in sublist])
    flat_train_label = np.array([item for sublist in train_results['label'] for item in sublist])
    
    flat_val_pred = np.array([item for sublist in valid_results['pred'] for item in sublist])
    flat_val_label = np.array([item for sublist in valid_results['label'] for item in sublist])

    ## Calculate performance metrics    
    #F1 score
    train_acc = metrics.f1_score(flat_train_label, np.where(flat_train_pred > 0.5, 1, 0))
    val_acc = metrics.f1_score(flat_val_label, np.where(flat_val_pred > 0.5, 1, 0))
    train_accuracy.append(train_acc) # running score
    valid_accuracy.append(val_acc) # running score

    # ROC AUC
    train_fpr, train_tpr, _ = metrics.roc_curve(flat_train_label, flat_train_pred)
    val_fpr, val_tpr, _ = metrics.roc_curve(flat_val_label, flat_val_pred)
    train_roc_auc = round(metrics.auc(train_fpr, train_tpr), 5)
    val_roc_auc = round(metrics.auc(val_fpr, val_tpr), 5)
    running_train_roc.append(train_roc_auc)
    running_val_roc.append(val_roc_auc)

    # Caclulate PRC AUC
    train_precision, train_recall, train_thresholds = metrics.precision_recall_curve(flat_train_label, flat_train_pred)
    train_prc_auc = round(metrics.auc(train_recall, train_precision), 5)
    val_precision, val_recall, val_thresholds = metrics.precision_recall_curve(flat_val_label, flat_val_pred)
    val_prc_auc = round(metrics.auc(val_recall, val_precision), 5)
    running_train_prc.append(train_prc_auc)
    running_val_prc.append(val_prc_auc)
 
    # Print epoch results
    epoch_lr = optimizer.param_groups[0]["lr"]
    learning_rate.append(epoch_lr)
    print(f'\nEpoch {epoch+1} learning rate: {epoch_lr}')
    print(f'Train Loss: {train_results["train_loss"]:.4f}')
    print(f'Val Loss: {valid_results["val_loss"]:.4f}\n')
    print(f'Training F1 accuracy = {train_acc:.5f}')
    print(f'Validation F1 accuracy = {val_acc:.5f}\n')

    print(f'Training ROC_auc: {train_roc_auc}')
    print(f'Validation ROC_auc: {val_roc_auc}\n')

    print(f'Training PRC_auc: {train_prc_auc}')
    print(f'Validation PRC_auc: {val_prc_auc}\n')
    print('\n------------------\n')

    #Plot and save figures on performance metrics over each epoch
    plot_loss_f1(train_loss, valid_loss, train_accuracy, valid_accuracy)
    plot_roc_prc(running_train_roc, running_val_roc, running_train_prc, running_val_prc)

    ## Save metrics to csv
    epoch_metrics = pd.DataFrame(data = {'epoch':list(range(epoch + 1)),'learning_rate':learning_rate,
                                        'train_loss': train_loss, 'train_f1': train_accuracy,
                                        'valid_loss':valid_loss, 'valid_f1':valid_accuracy,
                                        'train_roc_auc': running_train_roc, 'val_roc_auc':running_val_roc,
                                        'train_prc_auc':running_train_prc, 'val_prc_auc': running_val_prc})
    epoch_metrics.to_csv(f'../output/{now.strftime("%b-%d-%Y")}_epoch_metrics.csv', index = False)

    # Save trained model to disk
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f'../output/{epoch}_model.pt')
    torch.cuda.empty_cache()

sys.stdout.close()
#fin