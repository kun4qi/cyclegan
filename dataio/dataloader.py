from .transformation import Normalize
from .transformation import ZScoreNormalize
from .transformation import ToImage
from .transformation import ToTensor
from .transformation import RandomHorizontalFlip
from .transformation import RandomVerticalFlip
from .transformation import RandomRotate
from .transformation import RandomScale
from .transformation import RandomColorJitter
from .transformation import RandomSliceSelect

import numpy as np
import os
import re
from glob import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pytorch_lightning import LightningDataModule

class CKBrainMetDataset(Dataset):

    def __init__(self, config, mode, normal_patient_paths, abnormal_patient_paths, transform, image_size):
        super().__init__()
        assert mode in ['train', 'test']
        """
        if mode == train       -> output only normal images without label
        if mode == test        -> output both normal and abnormal images with label
        """
        self.config = config
        self.mode = mode
        self.normal_patient_paths = normal_patient_paths
        self.abnormal_patient_paths = abnormal_patient_paths
        self.transform = transform
        self.image_size = image_size
        self.normal_files = self.build_file_paths(self.normal_patient_paths)
        self.abnormal_files = self.build_file_paths(self.abnormal_patient_paths)

    def build_file_paths(self, patient_paths):
        files = []
        for patient_path in patient_paths:
            file_paths = glob(os.path.join(patient_path + "/*" + self.config.dataset.select_slice + ".npy")) #指定のスライスのパスを取得
            for file_path in file_paths:
                if self.mode == 'train':
                    files.append({
                        'image': file_path
                    })

                elif self.mode == 'test' or self.mode == 'test_normal':
                    label_path = self.get_label_path(file_path)

                    files.append({
                        'image': file_path,
                        'label': label_path
                    })

        return files

    def get_label_path(self, file_path):
        file_path = file_path.replace(self.config.dataset.select_slice, 'seg')
        return file_path

    def __len__(self):
        return max(len(self.normal_files), len(self.abnormal_files))


    def __getitem__(self, index):
        normal_image = np.load(self.normal_files[index % len(self.normal_files)]['image'])
        normal_image = np.flipud(np.transpose(normal_image))

        abnormal_image = np.load(self.abnormal_files[index % len(self.abnormal_files)]['image'])
        abnormal_image = np.flipud(np.transpose(abnormal_image))

        sample = {
            'normal_image': normal_image.astype(np.float32),
            'abnormal_image': abnormal_image.astype(np.float32)
        }

        if self.mode == 'test':
            if os.path.exists(self.normal_files[index % len(self.normal_files)]['label']):
                normal_label = np.load(self.normal_files[index % len(self.normal_files)]['label'])
                normal_label = np.flipud(np.transpose(normal_label))
            else:
                normal_label = np.zeros_like(normal_image)

            if os.path.exists(self.abnormal_files[index % len(self.abnormal_files)]['label']):
                abnormal_label = np.load(self.abnormal_files[index % len(self.abnormal_files)]['label'])
                abnormal_label = np.flipud(np.transpose(abnormal_label))
            else:
                abnormal_label = np.zeros_like(abnormal_image)

            sample.update({
                'normal_label': normal_label.astype(np.int32),
                'abnormal_label': abnormal_label.astype(np.int32)
            })

        if self.transform:
            sample = self.transform(sample)

        return sample


class CKBrainMetDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.root_dir_path = self.config.dataset.root_dir_path
        self.CKBrainMetDataset = CKBrainMetDataset
        self.omit_transform = False

    def get_patient_paths(self, base_dir_path):
        patient_ids = os.listdir(base_dir_path)
        return [os.path.join(base_dir_path, p) for p in patient_ids]

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            
            if self.config.dataset.use_augmentation:
                transform = transforms.Compose([
                    ToImage(),
                    RandomHorizontalFlip(),
                    RandomRotate(degree=20),
                    RandomScale(mean=1.0, var=0.05, image_fill=0),
                    # RandomColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    ToImage(),
                    ToTensor(),
                ])

            val_transform = transforms.Compose([
                    ToImage(),
                    ToTensor(),
                ])

            if self.omit_transform:
                transform = None
            
            normal_train_patient_paths = self.get_patient_paths(os.path.join(self.root_dir_path, 'MICCAI_BraTS_2019_Data_Val_Testing/Normal'))
            abnormal_train_patient_paths = self.get_patient_paths(os.path.join(self.root_dir_path, 'MICCAI_BraTS_2019_Data_Val_Testing/Abnormal'))
            self.train_dataset = self.CKBrainMetDataset(config=self.config, mode='train', normal_patient_paths=normal_train_patient_paths, abnormal_patient_paths=abnormal_train_patient_paths, transform=transform, image_size=self.config.dataset.image_size)
            self.valid_dataset = self.CKBrainMetDataset(config=self.config, mode='train', normal_patient_paths=normal_train_patient_paths, abnormal_patient_paths=abnormal_train_patient_paths, transform=val_transform, image_size=self.config.dataset.image_size)
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            transform = transforms.Compose([
                    ToImage(),
                    ToTensor(),
                    Normalize(min_val=0, max_val=255),
                ])
            normal_test_patient_paths = self.get_patient_paths(os.path.join(self.root_dir_path, 'MICCAI_BraTS_2019_Data_Training/Normal'))
            abnormal_test_patient_paths = self.get_patient_paths(os.path.join(self.root_dir_path, 'MICCAI_BraTS_2019_Data_Training/Abnormal'))
            self.test_dataset = self.CKBrainMetDataset(config=self.config, mode='test', normal_patient_paths=normal_test_patient_paths, abnormal_patient_paths=abnormal_test_patient_paths, transform=transform, image_size=self.config.dataset.image_size)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.dataset.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.config.dataset.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.dataset.batch_size, shuffle=False)