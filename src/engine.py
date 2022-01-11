# Engine for training
import torch
from tqdm import tqdm

#Training fxn
def train(model, dataloader, optimizer, criterion, train_data, device):
  print('training...')
  model.train()
  counter = 0
  train_running_loss = 0.0
  train_running_preds = []
  train_running_labels = []
  for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
    counter += 1 
    image, target = data['image'].to(device), data['label'].to(device)
    optimizer.zero_grad()
    outputs = model(image)
    sigmoid_outputs = torch.sigmoid(outputs)
    # Apply sigmoid activation so outputs are b/t 0 and 1
    #outputs = torch.sigmoid(outputs) ## commented out with nn.BCEWithLogitsLoss as criterion instead
    loss = criterion(outputs, target)
    train_running_loss += loss.item()

    train_running_preds += sigmoid_outputs.tolist()
    train_running_labels += target.tolist()
    #backprop
    loss.backward()
    #update optimizer parameters
    optimizer.step()
  scheduler.step()
  # Computes training loss by dividing total loss by number of batches
  train_loss = train_running_loss / counter
  return {'train_loss': train_loss, 'pred':train_running_preds, 'label':train_running_labels}

  # Validation fxn
def validate(model, dataloader, criterion, val_data, device):
  print('validating...')
  model.eval()
  counter = 0
  val_running_loss = 0.0
  val_running_preds = []
  val_running_labels = []
  with torch.no_grad():
    for i, data in tqdm(enumerate(dataloader), total = int(len(val_data)/dataloader.batch_size)):
      counter += 1
      image, target = data['image'].to(device), data['label'].to(device)
      outputs = model(image)
      sigmoid_outputs = torch.sigmoid(outputs)
      # apply sigmoid activation to get outputs b/t 0 and 1
      #outputs = torch.sigmoid(outputs)
      loss = criterion(outputs, target)
      val_running_loss += loss.item()
      #binary_outputs = torch.where(sigmoid_outputs > 0.5, 1, 0)
      val_running_preds += sigmoid_outputs.tolist()
      val_running_labels += target.tolist()
    val_loss = val_running_loss / counter

    return {'val_loss': val_loss, 'pred':val_running_preds, 'label':val_running_labels}