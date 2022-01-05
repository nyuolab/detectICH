from torchvision import models as models
import torch.nn as nn

def model_fxn(pretrained, requires_grad):
# use ResNeXt101 model with ImageNet pretrained weights
  model = models.resnext101_32x8d(progress = True, pretrained=pretrained)
  # to freeze the hidden layers
  if requires_grad == False:
    for param in model.parameters():
        param.requries_grad = False
      # to train the hidden layers
  elif requires_grad == True:
    for param in model.parameters():
        param.requires_grad = True
  # make the classification layer learnable
  # we have 6 classes in total
  model.fc = nn.Linear(2048, 6)
  return model