from cgi import test
import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset_class import IntracranialDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Point to the relevent test label data and DICOM files
data_path = '../holdout_test_data' # folder containing dicom images
if os.path.exists('../inf_output'):
    print('inference output path exists')
else:
    print("\noutput directory does not exist")
    os.mkdir('../inf_output/')
    print("Directory ../inf_output/ created\n")

#Inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize model
model_inf = models.model_fxn(pretrained = False, requires_grad = False).to(device)


#
model_inf = torch.nn.DataParallel(model_inf).to(device)


# load model checkpoint
checkpoint = torch.load('../output/model.pt')
# load model weights state_dict
#model_inf.load_state_dict(checkpoint['model_state_dict'])
model_inf.load_state_dict(checkpoint['model']) # if using FL model
model_inf.eval()


#
label_options = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']

test_data = IntracranialDataset(path=data_path, train=False, test=True)

test_loader = DataLoader(
    test_data,
    batch_size = 512,
    shuffle=False,
    num_workers=8
)

samples = []
pred_labels = []
ground_truth = []
for counter, data in tqdm(enumerate(test_loader), total=int(len(test_data)/test_loader.batch_size)):
    image, target = data['image'].to(device), data['label']
    # get all the index positions where value == 1
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    # get the predictions by passing the image through the model
    outputs = model_inf(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    samples.append(data['sample_id'])
    pred_labels.append(outputs.numpy())
    ground_truth.append(target.numpy())

predictions_df = pd.melt(pd.concat(
    [pd.DataFrame(np.concatenate(samples), columns = ['Image']),
     pd.DataFrame(np.concatenate(pred_labels), columns = label_options)],
     axis=1, join = 'inner'
), id_vars = ['Image'], var_name = 'subtype', value_name = 'pred_label')


ground_truth_df = pd.melt(pd.concat(
    [pd.DataFrame(np.concatenate(samples), columns = ['Image']),
     pd.DataFrame(np.concatenate(ground_truth), columns = label_options)],
     axis=1, join = 'inner'
), id_vars = ['Image'], var_name = 'subtype', value_name = 'ground_truth')

# Save a .csv file with the inference results (label predictions) and ground-truth labels
pd.merge(predictions_df, ground_truth_df).to_csv('../inf_output/inference_results.csv', index = False)

# Format and save .csv with ID and predicted labels for Kaggle submission
#kaggle_submission = pd.merge(predictions_df, ground_truth_df)
#kaggle_submission['ID'] = kaggle_submission['Image'] + '_' + kaggle_submission['subtype']
#kaggle_submission = kaggle_submission[['ID', 'pred_label']].rename(columns = {'pred_label': 'Label'})
#kaggle_submission.to_csv('../output/kaggle_submission.csv', index = False)