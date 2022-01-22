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

import os.path
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose

# From train.py --probably can trim this down
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from fl_dataset_class import IntracranialDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from datetime import date
from sklearn import metrics
from tqdm import tqdm

#
from nvflare.apis.dxo import from_shareable, DXO, DataKind, MetaKey
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from pt_constants import PTConstants
from resnext_network import MyResNeXtClass#,model_fxn
from resnext_class import ResNet, Bottleneck

class ICHTrainer(Executor):

    def __init__(self, lr=0.01, epochs=5, train_task_name=AppConstants.TASK_TRAIN,
                 submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL, exclude_vars=None):
        """Cifar10 Trainer handles train and submit_model tasks. During train_task, it trains a
        simple network on CIFAR10 dataset. For submit_model task, it sends the locally trained model
        (if present) to the server.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
        """
        super(ICHTrainer, self).__init__()

        #
        self._lr = lr
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars

        # Training setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.model = model_fxn(pretrained=True, requires_grad = False)
        self.model = ResNet(
            Bottleneck,
            layers=[3, 4, 23, 3],
            groups = 32,
            width_per_group = 8,
            pretrained = True,
        )
        self.model.to(self.device)
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        batch_size = 8
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size = 3, gamma=0.1)


        # Create Cifar10 dataset for training.
        #transforms = Compose([
        #    ToTensor(),
        #    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #])
        # Point to the relevent test label data and DICOM files
        train_csv = pd.read_csv('./input/label_dataset/prototype_train_labels.csv')
        data_path = './input/images'

        #self._train_dataset = CIFAR10(root='~/data', transform=transforms, download=True, train=True)
        self._train_dataset = IntracranialDataset(train_csv, path=data_path,train=True,test=False)
        self._train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True)
        self._n_iterations = len(self._train_loader)

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf)

    def local_train(self, fl_ctx, weights, abort_signal):
        print('training...')
        # Set the model weights
        self.model.load_state_dict(state_dict=weights)

        # Basic training
        self.model.train()
        for epoch in range(self._epochs):
            print(f'Epoch {epoch+1} of {self._epochs}')
            running_loss = 0.0
            for i, batch in tqdm(enumerate(self._train_loader),total=int(len(train_data)/dataloader.batch_size)):
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                images, labels = batch['image'].to(self.device), batch['label'].to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(images)
                cost = self.loss(predictions, labels)
                cost.backward()
                self.optimizer.step()

                # Ask jiahui significance of 3000? b/c CIFAR10 has 60,000 images?
                running_loss += (cost.cpu().detach().numpy()/images.size()[0])
                if i % 3000 == 0:
                    self.log_info(fl_ctx, f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, "
                                          f"Loss: {running_loss/3000}")
                    running_loss = 0.0
            self.scheduler.step()

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                self.local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self.save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                new_weights = self.model.state_dict()
                new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

                outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_weights,
                                   meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations})
                return outgoing_dxo.to_shareable()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self.load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except:
            self.log_exception(fl_ctx, f"Exception in simple trainer.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(data=torch.load(model_path),
                                                                   default_train_conf=self._default_train_conf)
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml
