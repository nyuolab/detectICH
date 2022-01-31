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
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import os

# From train.py --probably can trim this down
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from fl_engine import train, validate # I don't think I need this...
from fl_dataset_class import IntracranialDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from datetime import date
from sklearn import metrics
from tqdm import tqdm

#
from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
#from resnext_network import MyResNeXtClass #,model_fxn, 
from resnext_class import ResNet, Bottleneck
from simple_network import SimpleNetwork
from monai.networks.nets.senet import SENet
from monai.networks.blocks.squeeze_and_excitation import SEResNeXtBottleneck
from monai.networks.nets.unet import UNet

class ICHValidator(Executor):
    
    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION):
        super(ICHValidator, self).__init__()

        self._validate_task_name = validate_task_name

        # Set up the model
        self.model = ResNet(
            Bottleneck,
            layers=[3, 4, 23, 3],
            groups = 32,
            width_per_group = 8,
        )
        #self.model = SimpleNetwork()

        print(type(self.model))
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        
        # Point to the relevent test label data and DICOM files
        train_csv = pd.read_csv('./input/label_dataset/prototype_train_labels.csv')
        data_path = './input/images'
        self.test_data = IntracranialDataset(train_csv, path=data_path, train=False, test=False)
        self.test_loader = DataLoader(self.test_data, batch_size=8, shuffle=False)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self.do_validation(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(fl_ctx, f"Accuracy when validating {model_owner}'s model on"
                                      f" {fl_ctx.get_identity_name()}"f's data: {val_accuracy}')

                dxo = DXO(data_kind=DataKind.METRICS, data={'val_acc': val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def do_validation(self, weights, abort_signal):
        self.model.load_state_dict(weights)
        self.model.eval()

        correct = 0
        total = 0
        # make running array of outputs for f1_score after all data
        running_output = []
        running_label = []
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                if abort_signal.triggered:
                    return 0

                images, labels = data['image'].to(self.device), data['label'].to(self.device)
                output = torch.sigmoid(self.model(images))
                bin_output = np.where(output > 0.5, 1, 0)
                #_, pred_label = torch.max(output, 1)
                #print(f"pred_label: {pred_label}")
                #correct += (pred_label == labels).sum().item()
                #total += images.size()[0]
                running_label.append(np.array(labels).flatten())
                running_output.append(np.array(bin_output).flatten())

            #metric = correct/float(total)
            flat_label = np.array([item for sublist in running_label for item in sublist])
            flat_output = np.array([item for sublist in running_output for item in sublist])
            metric = metrics.f1_score(flat_label, flat_output)
            print(f"f1 metric = {metric}")

        return metric
