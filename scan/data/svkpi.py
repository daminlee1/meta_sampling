"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.mypath import MyPath
from torchvision import transforms as tf
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from os import listdir
from os.path import isfile, join
import cv2

class SVKPI_V2(Dataset):
    """`SVKPI _ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'svkpi-batches-py'
    url = ""
    filename = "svkpi-batches-py.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = []
    train_dir = "/data"
    valid_dir = "/data"
    train_txt = "all.txt"
    valid_txt = "all.txt"
    file_dir = ""
    file_txt = ""
    imgs = []

    test_list = []
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root=MyPath.db_root_dir('svkpi'), train=True, transform=None, 
                    download=False, to_neighbors_dataset=False):

        super(SVKPI_V2, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
        self.to_neighbors_dataset = to_neighbors_dataset
        if download:
            self.download()

        if self.train:
            self.file_dir = self.train_dir+"/JPEGImages/"
            self.file_txt = self.train_dir+"/ImageSets/"+self.train_txt
        else:
            self.file_dir = self.valid_dir+"/JPEGImages/"
            self.file_txt = self.valid_dir+"/ImageSets/"+self.valid_txt

        #read file names in folder
        img_names = []
        img_data = []
        with open(self.file_txt, 'r', encoding='UTF-8') as f:
            img_names = f.readlines()
        for n in img_names:
            n = n.replace("\n","")
            if os.path.exists(self.file_dir + n + ".jpg"):
                img_data.append(self.file_dir + n + ".jpg")
            elif os.path.exists(self.file_dir + n + ".png"):
                img_data.append(self.file_dir + n + ".png")
        self.imgs = img_data
        print(len(self.imgs))
        self.resize = tf.Resize(608)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img_path = self.imgs[index]
        with open(img_path, 'rb') as f:
            img = np.array(Image.open(f).convert('RGB'), dtype=np.uint8)
        target = 255 #unknown

        #cv2.imwrite("color_converting"+str(index)+".jpg",img2)
        img = cv2.resize(img, (960,640), interpolation=cv2.INTER_LINEAR)

        img_size = img.size
        
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        img_name = img_path.replace(self.file_dir,"")
        
        if self.to_neighbors_dataset:
            out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'img_name': img_name}}
        else:
            out = img, target, {'im_size': img_size, 'index': index, 'img_name': img_name}
        
        return out

    def get_image(self, index):
        img_path = self.imgs[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB') 
        img = self.resize(img)
        return img
        
    def __len__(self):
        return len(self.imgs)

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")