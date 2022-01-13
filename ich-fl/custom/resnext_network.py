# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torchvision import models as tvmodels
import torch.nn as nn
import torch



def model_fxn(pretrained, requires_grad):
# use ResNeXt101 model with ImageNet pretrained weights
  model = tvmodels.resnext101_32x8d(progress = True, pretrained=pretrained)
  # to freeze the hidden layers
  if requires_grad == False:
    for param in model.parameters():
        param.requires_grad = False
  # to train the hidden layers
  elif requires_grad == True:
    for param in model.parameters():
        param.requires_grad = True
  # make the classification layer learnable
  # we have 6 classes in total
  model.fc = nn.Linear(2048, 6)
  return model