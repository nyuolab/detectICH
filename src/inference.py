import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset_class import IntracranialDataset
from torch.utils.data import DataLoader

# Point to the relevent test label data and DICOM files
testing_data = pd.read_csv('../input/label_dataset/prototype_test_labels.csv')
data_path = '../input/images'

#Inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize model
model_inf = models.model_fxn(pretrained = False, requires_grad = False).to(device)
# load model checkpoint
checkpoint = torch.load('../output/model.pth')
# load model weights state_dict
model_inf.load_state_dict(checkpoint['model_state_dict'])
model_inf.eval()


#
label_options = testing_data.columns.values[1:-1]

test_data = IntracranialDataset(
    testing_data, path = data_path, train = False, test = True
)

test_loader = DataLoader(
    test_data,
    batch_size = 1,
    shuffle=False
)

for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']
    # get all the index positions where value == 1
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    # get the predictions by passing the image through the model
    outputs = model_inf(image)
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    sorted_indices = np.argsort(outputs[0])
    # Keep top 6 (in this case, all)
    best = sorted_indices[-6:]
    string_predicted = ''
    string_actual = ''
    for i in range(len(best)):
        string_predicted += f"{label_options[best[i]]},{outputs[0][best[i]]:.5f}    "
    for i in range(len(target_indices)):
        string_actual += f"{label_options[target_indices[i]]}    "
    image = image.squeeze(0)
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
    plt.savefig(f"../output/inference_{counter}.jpg")