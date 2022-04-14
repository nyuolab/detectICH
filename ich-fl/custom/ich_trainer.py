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

import enum
import os.path
from random import shuffle
from ssl import AlertDescription
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import models as tvmodels
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
import matplotlib.pyplot as plt


class ICHTrainer(Executor):

    def __init__(self, lr=0.0003, epochs=2, train_task_name=AppConstants.TASK_TRAIN,
                 submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL, exclude_vars=None):
        """
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
        self.model = tvmodels.resnext101_32x8d(pretrained=True, progress=True)
        self.model.fc = nn.Linear(2048,6)
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        batch_size = 16
        #self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size = 3, gamma=0.1)


        # Point to the relevent test label data and DICOM files
        train_csv = pd.read_csv('./input/labels.csv')
        train_csv = train_csv.sample(frac=1, random_state=23)

        data_path = './input/data'

        self._train_dataset = IntracranialDataset(train_csv, path=data_path,train=True,test=False)
        self._train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True)
        self._n_iterations = len(self._train_loader)

        # Run validation
        self._val_dataset = IntracranialDataset(train_csv, path=data_path, train=False, test=False)
        self._val_loader = DataLoader(self._val_dataset, batch_size=batch_size, shuffle=False)

        # Define weighted loss for imabalanced dataset
        loss_weights = self._train_dataset.loss_weights.to(self.device)
        print(f'\nloss weights: {loss_weights}\n')
        self.loss = nn.BCEWithLogitsLoss(pos_weight=loss_weights)


        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf)

    def local_train(self, fl_ctx, weights, abort_signal):
        print('\ntraining...\n')
        # Set the model weights
        self.model.load_state_dict(state_dict=weights)
        # Basic training
        self.model.train()

        # Initialize variables to output
        running_train_loss = []
        running_val_loss = []
        running_train_acc = []
        running_val_acc = []
        running_train_f1 = []
        running_val_f1 = []
        running_train_roc = []
        running_val_roc = []
        running_train_prc = []
        running_val_prc = []
        #
        local_output_dir = self.create_output_dir(fl_ctx)
        for epoch in range(self._epochs):
            print(f'Epoch {epoch+1} of {self._epochs}')
            # running_loss = 0.0
            
            train_running_batch_loss = 0.0
            train_epoch_preds = []
            train_epoch_labels = []
            counter = 0
            for i, batch in tqdm(enumerate(self._train_loader),total=int(len(self._train_dataset)/self._train_loader.batch_size)):
                counter += 1
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                # Read in images and labels from batch for training
                images, labels = batch['image'].to(self.device), batch['label'].to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(images)
                cost = self.loss(predictions, labels)
                cost.backward()
                self.optimizer.step()
                sigmoid_preds = torch.sigmoid(predictions)

                # Add results (loss, predictions, labels) per batch to running lists
                train_running_batch_loss += cost.item()
                train_epoch_labels += labels.tolist()
                train_epoch_preds += sigmoid_preds.tolist()

                # running_loss += (cost.cpu().detach().numpy()/images.size()[0])
                # if i % 3000 == 0:
                #     self.log_info(fl_ctx, f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, "
                #                           f"Loss: {running_loss/3000}")
                #     running_loss = 0.0

            # Divide total loss added by num_batches
            train_epoch_loss = train_running_batch_loss / counter

            # Validate updated model for this epoch
            local_val_results = self.local_val(fl_ctx, abort_signal)

            # Flatten the results
            # Flatten labels and predictions for
            flat_train_pred = np.array([item for sublist in train_epoch_preds for item in sublist])
            flat_train_label = np.array([item for sublist in train_epoch_labels for item in sublist])
            
            flat_val_pred = np.array([item for sublist in local_val_results['preds'] for item in sublist])
            flat_val_label = np.array([item for sublist in local_val_results['labels'] for item in sublist])

            # Calculate metrics, run
            # Determine threshold to binarize outputs for accuracy/F1 score
            bin_threshold = 0.4
            #
            train_acc = metrics.accuracy_score(flat_train_label, np.where(flat_train_pred > bin_threshold, 1, 0))
            val_acc = metrics.accuracy_score(flat_val_label, np.where(flat_val_pred > bin_threshold, 1, 0))
            #
            train_f1 = metrics.f1_score(flat_train_label, np.where(flat_train_pred > bin_threshold, 1, 0))
            val_f1 = metrics.f1_score(flat_val_label, np.where(flat_val_pred > bin_threshold, 1, 0))
            # ROC AUC
            train_fpr, train_tpr, _ = metrics.roc_curve(flat_train_label, flat_train_pred)
            val_fpr, val_tpr, _ = metrics.roc_curve(flat_val_label, flat_val_pred)
            train_roc_auc = round(metrics.auc(train_fpr, train_tpr), 6)
            val_roc_auc = round(metrics.auc(val_fpr, val_tpr), 6)

             # Caclulate PRC AUC
            train_precision, train_recall, train_thresholds = metrics.precision_recall_curve(flat_train_label, flat_train_pred)
            train_prc_auc = round(metrics.auc(train_recall, train_precision), 6)
            val_precision, val_recall, val_thresholds = metrics.precision_recall_curve(flat_val_label, flat_val_pred)
            val_prc_auc = round(metrics.auc(val_recall, val_precision), 6)
            
            print("\n++++++++++++++++++++++++")
            print(f'training accuracy: {train_acc}')
            print(f'validation accuracy: {val_acc}')

            # Append results from this epoch to running list
            running_train_loss.append(train_epoch_loss)
            running_val_loss.append(local_val_results["val_loss"])

            running_train_acc.append(train_acc)
            running_val_acc.append(val_acc)

            running_train_f1.append(train_f1)
            running_val_f1.append(val_f1)

            running_train_roc.append(train_roc_auc)
            running_val_roc.append(val_roc_auc)

            running_train_prc.append(train_prc_auc)
            running_val_prc.append(val_prc_auc)
            print(f'\nEpoch {epoch+1}:')
            print(f'Train Loss: {train_epoch_loss:.4f}')
            print(f'Val Loss: {local_val_results["val_loss"]:.4f}\n')
            
            print(f'Training accuracy = {train_acc:.5f}')
            print(f'Validation accuracy = {val_acc:.5f}\n')

            print(f'Training F1 score = {train_f1:.5f}')
            print(f'Validation F1 score = {val_f1:.5f}\n')

            print(f'Training ROC_auc: {train_roc_auc}')
            print(f'Validation ROC_auc: {val_roc_auc}\n')

            print(f'Training PRC_auc: {train_prc_auc}')
            print(f'Validation PRC_auc: {val_prc_auc}\n')

            # Plot metrics
            self.plot_metrics(fl_ctx, local_output_dir, running_train_loss, running_val_loss, 'Loss_per_epoch')
            self.plot_metrics(fl_ctx, local_output_dir, running_train_acc, running_val_acc, 'Accuracy_per_epoch')
            self.plot_metrics(fl_ctx, local_output_dir, running_train_f1, running_val_f1, 'F1_per_epoch')
            self.plot_metrics(fl_ctx, local_output_dir, running_train_roc, running_val_roc, 'ROC_auc_per_epoch')
            self.plot_metrics(fl_ctx, local_output_dir, running_train_prc, running_val_prc, 'Precision-Recall_auc_per_epoch')

            ## Save metrics to csv 
            print(f'saving to {local_output_dir}/epoch_metrics.csv')
            epoch_metrics = pd.DataFrame(data = {'epoch':list(range(epoch + 1)),
                                        'train_loss': running_train_loss, 'train_f1': running_train_f1,
                                        'valid_loss':running_val_loss, 'valid_f1':running_val_f1,
                                        'train_roc_auc': running_train_roc, 'val_roc_auc':running_val_roc,
                                        'train_prc_auc':running_train_prc, 'val_prc_auc': running_val_prc})
            
            epoch_metrics.to_csv(f'{local_output_dir}/epoch_metrics.csv', index = False)
    
    def local_val(self, fl_ctx, abort_signal):
        print('\nvalidating...\n')
        self.model.eval()
        val_running_loss = 0.0
        val_running_preds = []
        val_running_labels = []
        counter = 0
        with torch.no_grad():
            for i, batch, in tqdm(enumerate(self._val_loader), total=int(len(self._val_dataset)/self._val_loader.batch_size)):
                if abort_signal.triggered:
                    return
                counter += 1
                images, labels = batch['image'].to(self.device), batch['label'].to(self.device)
                predictions = self.model(images)
                cost = self.loss(predictions,labels)
                sigmoid_preds = torch.sigmoid(predictions)
                val_running_loss += cost.item()
                val_running_labels += labels.tolist()
                val_running_preds += sigmoid_preds.tolist()

            val_loss = val_running_loss / counter
        return {'val_loss':val_loss, 'preds':val_running_preds, 'labels':val_running_labels}


    def plot_metrics(self, fl_ctx, output_dir, train_data, val_data, title_string):
        # Plot
        fig, axs = plt.subplots(figsize = (10,7))
        axs.plot(train_data, color = 'orange', label='train results')
        axs.plot(val_data, color = 'red', label='validation results')
        axs.set_title(f"{title_string}")
        axs.legend(loc='center left')
        print(f"plotting {title_string}...")
        #run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        #local_output_dir = os.path.join(run_dir, PTConstants.OutputMetricsDir)
        fig.savefig(f'{output_dir}/{title_string}.png')
        return

    #def calculate_metrics():
        # F1
        # Acc
        # ROC AUC
        # PRC AUC
    #    return


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


    def create_output_dir(self, fl_ctx: FLContext):
        # Make directory named "output" in current run_# folder
        # Get run number dir path
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        output_dir = os.path.join(run_dir, PTConstants.OutputMetricsDir)
        # make output folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        round_int = 1
        round_output_dir = os.path.join(run_dir, PTConstants.OutputMetricsDir, f"round_{round_int}")
        while os.path.exists(round_output_dir):
            round_int += 1
            round_output_dir = os.path.join(run_dir, PTConstants.OutputMetricsDir, f"round_{round_int}")
        os.makedirs(round_output_dir)
        print('\ncreating output dir...')
        print(round_output_dir)
        return round_output_dir


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
