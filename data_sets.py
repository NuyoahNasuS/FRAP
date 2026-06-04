import gzip
import pickle
import torchvision
import numpy as np
import os
from PIL import Image
import torch
import urllib

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR100_CLASSES = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

class CIFAR10v2(torchvision.datasets.CIFAR10):
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False):
        self.transform = transform
        self.target_transform = target_transform

        if train:
            data = np.load(os.path.join(root, 'cifar102_train.npz'), allow_pickle=True)
        else:
            data = np.load(os.path.join(root, 'cifar102_test.npz'), allow_pickle=True)
        
        self.data = data["images"]
        self.targets = data["labels"]

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray(img)
        if self.transform is not None:
          img = self.transform(img)
        if self.target_transform is not None:
              target = self.target_transform(target)
	    
        return img, target
    
class CIFAR10_C(torchvision.datasets.CIFAR10):
    def __init__(self, root, data_type=None, severity=1, transform = None, target_transform = None, download = False):
        self.transform = transform
        self.target_transform = target_transform

        data = np.load(os.path.join(root, f"{data_type}.npy"))
        labels = np.load(os.path.join(root, "labels.npy"))

        self.data = data[(severity - 1)*10000:severity*10000]
        self.targets = labels[(severity - 1)*10000:severity*10000].astype(np.int_)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target
    
    
class CIFAR100_C(torchvision.datasets.CIFAR100):
    def __init__(self, root, data_type=None, severity=1, transform = None, target_transform = None, download = False):
        self.transform = transform
        self.target_transform = target_transform

        data = np.load(os.path.join(root, f"{data_type}.npy"))
        labels = np.load(os.path.join(root, "labels.npy"))

        self.data = data[(severity - 1)*10000:severity*10000]
        self.targets = labels[(severity - 1)*10000:severity*10000].astype(np.int_)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target
    
class USPS(torch.utils.data.Dataset):
    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.filename = 'usps.pkl'
        self.train = train
        self.transform = transform
        self.dataset_size = None

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found.  You can use download=True to download it")
        
        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        img, label = self.train_data[index], self.train_labels[index]
        img = Image.fromarray(img.squeeze().astype(np.int8), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        label = int(label)
        return img, label
    
    def __len__(self):
        return self.dataset_size
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.filename))
    
    def download(self):
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)     # robustness
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels




