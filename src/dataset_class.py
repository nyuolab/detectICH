import torch
import cv2
import numpy as np 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pydicom
import os

class IntracranialDataset(Dataset):
  def __init__(self, csv_file, path, train, test, img_size=(512,512,1)):
    self.csv = csv_file
    self.path = path
    self.train = train
    self.test = test
    self.img_size = img_size
    self.all_image_names = self.csv[:]['Image']
    self.all_labels = np.array(self.csv.drop(['Image', 'all_diagnoses'], axis=1))
    self.train_ratio = int(0.85 * len(self.csv))
    self.valid_ratio = len(self.csv) - self.train_ratio

    # Set training data images and labels
    if self.train == True:
      print(f"Number of training images: {self.train_ratio}")
      self.image_names = list(self.all_image_names[:self.train_ratio])
      self.labels = list(self.all_labels[:self.train_ratio])

      #define training transforms
      self.transform = transforms.Compose([
                                           transforms.ToTensor(),                                   
      ])
    # Set validation data
    elif self.train == False and self.test == False:
      print(f"Number of validation images: {self.valid_ratio}")
      # saves 15% of dataset for validation, except for 10 images used for test set
      self.image_names = list(self.all_image_names[-self.valid_ratio:])
      self.labels = list(self.all_labels[-self.valid_ratio:])
      #Define validation transforms
      self.transform = transforms.Compose([
                                           transforms.ToTensor()
      ])
    
    # Set test data
    elif self.test == True and self.train == False:
      print(f"Number of test images: {len(self.all_image_names)}")
      self.image_names = list(self.all_image_names)
      self.labels = list(self.all_labels)
      self.transform = transforms.Compose([
                                           transforms.ToTensor()
      ])

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, idx):
    img_path = os.path.join(self.path, self.image_names[idx] + '.dcm')
    image = _read(img_path, desired_size = (256, 256, 3)))
    # Transform images
    image = self.transform(image)
    targets = torch.from_numpy(self.labels[idx])
    sample_id = self.image_names[idx]
    return {'sample_id': sample_id,
            'image': image.detach().clone().float(),
            'label': targets.detach().clone().float()
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
