import torch
import cv2
import numpy as np 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pydicom
import pandas as pd
import os

class IntracranialDataset(Dataset):
  def __init__(self, path, train, test, img_size=(512,512,1)):
    #self.csv = csv_file
    self.path = path # path to folder containing all site data
    self.train = train
    self.test = test
    self.img_size = img_size

    # Define master lists for for-loop to append
    self.master_image_paths = []
    self.master_labels = []
    self.master_image_names = []

    # Loop over each site within all_sites directory
    site_dirs = os.listdir(f'{self.path}')
    for site in site_dirs:
      
      # Define path for each site's data
      site_path = os.path.join(self.path, site)

      if os.path.isfile(site_path):
        continue

      print(f'\nAccessing data from: {site_path}')
      labels_path = site_path + '/input/labels.csv'
      
      # read csv
      site_labels_csv = pd.read_csv(labels_path)
      
      # Extract image names and labels
      site_image_names = site_labels_csv[:]['Image']
      site_image_paths = site_path + '/input/data/' + site_image_names + '.dcm'
      site_labels = np.array(site_labels_csv.drop(['Image', 'all_diagnoses'], axis = 1))

      # Define index range for training and validation images
      train_ratio = int(0.85 * len(site_labels_csv))
      valid_ratio = len(site_labels_csv) - train_ratio

      # Split up data based on training or validation
      # Set training data images and labels
      if self.train == True:
        print(f"Number of training images from {site}: {train_ratio}")
        image_paths = list(site_image_paths[:train_ratio])
        labels = list(site_labels[:train_ratio])
        image_names = list(site_image_names[:train_ratio])
        #define training transforms
        self.transform = transforms.Compose([
                                            transforms.ToTensor(),                                   
        ])

      # Set validation data
      elif self.train == False and self.test == False:
        print(f"Number of validation images from {site}: {valid_ratio}")
        # saves 15% of dataset for validation, except for 10 images used for test set
        image_paths = list(site_image_paths[-valid_ratio:])
        labels = list(site_labels[-valid_ratio:])
        image_names = list(site_image_names[-valid_ratio:])
        #Define validation transforms
        self.transform = transforms.Compose([
                                            transforms.ToTensor()
        ])
      
      # Set test data
      elif self.test == True and self.train == False:
        print(f"Number of test images: {len(site_image_names)}")
        image_paths = list(site_image_paths)
        labels = list(self.all_labels)
        image_names = list(site_image_names)
        self.transform = transforms.Compose([
                                            transforms.ToTensor()
        ])

      # Append each site data and labels to master list
      self.master_image_names.extend(image_names)
      self.master_image_paths.extend(image_paths)
      self.master_labels.extend(labels)


    ## Calculate the proportion of different labels to weight the loss
    total_n_samples = np.array(self.master_labels).shape[0]
    print(f"total samples: {total_n_samples}")
    n_per_subtype = np.sum(self.master_labels, axis = 0)
    print(n_per_subtype)

    n_positives = n_per_subtype[0]
    n_negatives = total_n_samples - n_positives
    
    #
    self.neg_pos_ratio = n_negatives / total_n_samples
    print(f'neg:pos ratio for loss weight: {self.neg_pos_ratio}')


    if self.train == True:
      print(f'\nTotal training set contains {len(self.master_image_paths)} images')
      img_dict = {'Image': self.master_image_names}
      df = pd.DataFrame(img_dict)
      df.to_csv('../' + 'all_site_training_subset.csv')
    elif self.train == False and self.test == False:
      print(f'\nTotal validation set contains {len(self.master_image_paths)} images')

## Additional functions
  def __len__(self):
    return len(self.master_image_paths)

  def __getitem__(self, idx):
    img_path = self.master_image_paths[idx]
    image = _read(img_path, desired_size = (256, 256, 3))
    # Transform images
    image = self.transform(image)
    targets = torch.from_numpy(self.master_labels[idx])
    sample_id = self.master_image_names[idx]
    return {'sample_id': sample_id,
            'image': image.detach().clone().float(),
            'label': targets.detach().clone().float(),
            'loss_weights': self.neg_pos_ratio
      }

## Functions to Normalize DCM images
def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000
    return(dcm)

def window_image(dcm, window_center, window_width):
    
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img

def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)
    
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    return bsb_img

def _read(path, desired_size):
    dcm = pydicom.dcmread(path)
    try:
        img = bsb_window(dcm)
    except:
        img = np.zeros(desired_size)
    
    img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)
    
    return img
