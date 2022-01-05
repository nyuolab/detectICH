import torch
from tqdm import tqdm

#Training fxn
def train(model, dataloader, optimizer, criterion, train_data, device):
  print('Training...')
  model.train()
  counter = 0
  train_running_loss = 0.0
  for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
    counter += 1 
    image, target = data['image'].to(device), data['label'].to(device)
    optimizer.zero_grad()
    outputs = model(image.float())
    # Apply sigmoid activation so outputs are b/t 0 and 1
    #outputs = torch.sigmoid(outputs)
    # Do not need sigmoid function if using nn.BCEWithLogitsLoss as criterion instead
    loss = criterion(outputs, target)
    train_running_loss += loss.item()
    #backprop
    loss.backward()
    #update optimizer parameters
    optimizer.step()
    torch.cuda.empty_cache()

  train_loss = train_running_loss / counter
  return train_loss

# Validation fxn
def validate(model, dataloader, criterion, val_data, device):
  print('Validating...')
  model.eval()
  counter = 0
  val_running_loss = 0.0
  with torch.no_grad():
    for i, data in tqdm(enumerate(dataloader), total = int(len(val_data)/dataloader.batch_size)):
      counter += 1
      image, target = data['image'].to(device), data['label'].to(device)
      outputs = model(image.float())
      # Apply sigmoid activation so outputs are b/t 0 and 1
      #outputs = torch.sigmoid(outputs)
      # Do not need sigmoid function if using nn.BCEWithLogitsLoss as criterion instead
      loss = criterion(outputs, target)
      val_running_loss += loss.item()
      torch.cuda.empty_cache()

    val_loss = val_running_loss / counter
    return val_loss