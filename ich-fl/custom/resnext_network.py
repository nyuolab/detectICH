
import torch.nn as nn
import torch

class MyResNeXtClass(nn.Module):
  def __init__(self, pretrained, requires_grad):
    super(MyResNeXtClass, self).__init__()
    self.pretrained = pretrained
    self.requires_grad = requires_grad
    # use ResNeXt101 model with ImageNet pretrained weights
    self.model = tvmodels.resnext101_32x8d(progress = True, pretrained=pretrained)

    # to freeze the hidden layers
    if requires_grad == False:
      print("freezing hidden layers...")
      for param in self.model.parameters():
          param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
      print("training hidden layers...")
      for param in self.model.parameters():
          param.requires_grad = True
    self.model.fc = nn.Linear(2048, 6)

  def forward(self, inputs):
    outputs = self.model(inputs)
    return outputs

#def model_fxn(pretrained, requires_grad):
# use ResNeXt101 model with ImageNet pretrained weights
#  model = tvmodels.resnext101_32x8d(progress = True, pretrained=pretrained)
#  # to freeze the hidden layers
#  if requires_grad == False:
#    for param in model.parameters():
#        param.requires_grad = False
#  # to train the hidden layers
#  elif requires_grad == True:
#    for param in model.parameters():
#        param.requires_grad = True
#  # make the classification layer learnable
#  # we have 6 classes in total
#  model.fc = nn.Linear(2048, 6)
#  return model