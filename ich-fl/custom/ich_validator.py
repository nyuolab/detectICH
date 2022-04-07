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
from torchvision.transforms import Compose, ToTensor, Normalize
import os

import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fl_dataset_class import IntracranialDataset
from torch.utils.data import DataLoader
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
from torchvision import models as tvmodels

class ICHValidator(Executor):
    
    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION):
        super(ICHValidator, self).__init__()

        self._validate_task_name = validate_task_name

        # Set up the model
        self.model = tvmodels.resnext101_32x8d(pretrained=True, progress=True)
        self.model.fc = nn.Linear(2048,6)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        
        # Point to the relevent test label data and DICOM files
        train_csv = pd.read_csv('./input/labels.csv')
        data_path = './input/data'
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
                validation_results = self.do_validation(weights, abort_signal)
                any_results = validation_results['any']
                epidural_results = validation_results['epidural']
                intraparenchymal_results = validation_results['intraparenchymal']
                intraventricular_results = validation_results['intraventricular']
                subarachnoid_results = validation_results['subarachnoid']
                subdural_results = validation_results['subdural']

                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(fl_ctx, f"ROC_auc, PRC_auc, and accuracy when validating {model_owner}'s model on"
                                      f" {fl_ctx.get_identity_name()}"f's data: {any_results}')

                dxo = DXO(data_kind=DataKind.METRICS, data={'any': any_results, 'epidural': epidural_results, 'intraparenchymal': intraparenchymal_results,'intraventricular':intraventricular_results, 'subarachnoid':subarachnoid_results, 'subdural':subdural_results})
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
        running_outputs = None
        running_labels = None
        label_list = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                if abort_signal.triggered:
                    return 0

                images, labels = data['image'].to(self.device), data['label'].to(self.device)
                outputs = torch.sigmoid(self.model(images))

                if i == 0:
                    running_labels=labels.cpu()
                    running_outputs=outputs.cpu()
                else:
                    running_labels=torch.cat((running_labels, labels.cpu()))
                    running_outputs=torch.cat((running_outputs, outputs.cpu()))


        metrics_output = {}

        for i in range(len(label_list)):
            subtype_labels=np.array(running_labels[:, i]).flatten()
            subtype_outputs=np.array(running_outputs[:, i]).flatten()

            #
            fpr, tpr, _ = metrics.roc_curve(subtype_labels, subtype_outputs)
            roc_auc = metrics.auc(fpr, tpr)
            #
            precision, recall, thresholds = metrics.precision_recall_curve(subtype_labels, subtype_outputs)
            prc_auc = metrics.auc(recall, precision)

            acc = metrics.accuracy_score(running_labels, running_outputs)
            #bin_output = np.where(output.cpu() > 0.5, 1, 0)
            #_, pred_label = torch.max(output, 1)
            #print(f"pred_label: {pred_label}")
            #correct += (pred_label == labels).sum().item()
            #total += images.size()[0]

            #metric = correct/float(total)
            #flat_label = np.array([item for sublist in running_label for item in sublist])
            #flat_output = np.array([item for sublist in running_output for item in sublist])
            #print(flat_output)
            
            #metric = metrics.f1_score(flat_label, flat_output)
            #print(f"f1 metric = {metric}")

            metrics_output[label_list[i]]=(roc_auc, prc_auc, acc)
            print(f"\nResults of metrics are...\n")
            print(metrics_output)
        return metrics_output
