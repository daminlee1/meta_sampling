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

class SVKPI(Dataset):
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
    train_dir = "/damin/data/GODTest_SVKPI_3000km/ImageSets/all.txt"
    valid_dir = "/damin/data/GODTest_SVKPI_3000km/ImageSets/all.txt"
    file_dir = ""
    data_num = 0

    test_list = []
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root=MyPath.db_root_dir('svkpi'), train=True, transform=None, 
                    download=False):

        super(SVKPI, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.classes = ['unknown']

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            self.file_dir = self.train_dir
        else:
            self.file_dir = self.valid_dir

        self.data = []
        self.targets = []

        #read file names in folder
        img_names = []
        with open(self.file_dir, 'r') as f:
            img_names = f.readlines()
        file_list = [f for f in listdir(self.file_dir) if isfile(join(self.file_dir, f))]
        file_list = [f for f in file_list if ".png" in str(f) or ".jpg" in str(f)]
        for i,f in enumerate(file_list):
            if i % 2 != 0:
                continue
            img = Image.open(self.file_dir+f)
            img = img.resize((416, 416), resample=Image.BILINEAR)
            self.data.append(img)
            if i % 100 == 0:
                print("reading training data - {}/{}".format(i / 2, min(len(file_list), self.data_num)))
            if i == self.data_num*2:
                break
                

        self.data = np.vstack(self.data).reshape(-1, 3, 416, 416)
        print("self.data.shape:",self.data.shape)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.zeros(self.data.shape[0])
        #self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[int(target)]        

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
        
        return out

    def get_image(self, index):
        img = self.data[index]
        return img
        
    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            print(fentry[0],  fentry[1])
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

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
    train_dir = "/damin/data/GODTrain211111"
    valid_dir = "/damin/data/GODTrain211111"
    train_txt = "od_tstld_clean.txt"
    valid_txt = "od_tstld_clean.txt"
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
                    download=False):

        super(SVKPI_V2, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']

        if download:
            self.download()

        if self.train:
            print("train")
            self.file_dir = self.train_dir+"/JPEGImages/"
            self.file_txt = self.train_dir+"/ImageSets/"+self.train_txt
        else:
            print("valid")
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
            img = img[:,:,::-1]
            gray_img = np.array(Image.open(f).convert('L'), dtype=np.uint8)
        target = 255 #unknown
        # when APTIV data, crop bottom area in the image.
        img = img[:690,:,:]
        #cv2.imwrite("original"+str(index)+".jpg",img)
        gray_img = gray_img[:690,:]
        # Color converting (RGB -> RCCC likely). remain R value and convert B,G to gray.
        img2 = np.copy(img)
        # B -> gray
        img2[:,:,0] = gray_img[:,:]
        # G -> gray
        img2[:,:,1] = gray_img[:,:]
        #cv2.imwrite("color_converting"+str(index)+".jpg",img2)
        img = cv2.resize(img, (960,640), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (960,640), interpolation=cv2.INTER_LINEAR)
        #img = self.resize(img)
        img_size = img.size
        
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        img_name = img_path.replace(self.file_dir,"")
        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'img_name': img_name}}
        
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