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

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(387096, 120)
        self.fc2 = nn.Linear(120, 6)

    def forward(self, x):
        print(f"{x.shape}\n")
        x = self.pool(F.relu(self.conv1(x)))
        print(f"pre flatten: {x.shape}\n")
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        print(f"post flatten: {x.shape}\n")
        x = F.relu(self.fc1(x))
        print(f"post relu/fc1: {x.shape}\n")
        x = self.fc2(x)
        print(f"post fc2: {x.shape}\n")
        return x
