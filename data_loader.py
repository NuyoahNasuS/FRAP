import torchvision.transforms as transforms
import torch, torchvision
import torchvision.transforms.functional as TF
from transformers import BertTokenizerFast, DistilBertTokenizerFast
import os
from data_sets import *
from wilds.datasets.fmow_dataset import FMoWDataset
from wilds.datasets.rxrx1_dataset import RxRx1Dataset
from wilds.datasets.amazon_dataset import AmazonDataset
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.wilds_dataset import WILDSSubset


def get_transforms(data: str, net=None, max_token_length=None):
    if data in ["CIFAR10", "CIFAR10v2", "CIFAR10_C", "CIFAR100", "CIFAR100_C"]:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif data in ["MNIST", "QMNIST", "USPS"]:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transform_train
    elif data in ["SVHN"]:
        transform_train = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transform_train
    elif data.startswith("ImageNet") or data.startswith("Tiny_ImageNet"):
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_test = transform_train
    elif data in ["ImageNet_Breeds"]:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4717, 0.4499, 0.3837], [0.2600, 0.2516, 0.2575])
		])
        transform_test = transform_train
    elif data in ["Camelyon17", "Fmow"]:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
        transform_test = transform_train
    elif data in ["Rxrx1"]:
        def standardize(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=(1, 2))
            std = x.std(dim=(1, 2))
            std[std == 0.] = 1.
            return TF.normalize(x, mean, std)
        t_standardize = transforms.Lambda(lambda x: standardize(x))

        angles = [0, 90, 180, 270]
        def random_rotation(x: torch.Tensor) -> torch.Tensor:
            angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
            if angle > 0:
                x = TF.rotate(x, angle)
            return x
        t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

        transform_train_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
        transform_test_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
        transform_train = transforms.Compose(transform_train_ls)
        transform_test = transforms.Compose(transform_test_ls)
    elif data in ["Amazon", "CivilComments"]:
        def getBertTokenizer(model):
            if model == 'bert-base-uncased':
                tokenizer = BertTokenizerFast.from_pretrained(model)
            elif model == 'distilbert-base-uncased':
                tokenizer = DistilBertTokenizerFast.from_pretrained(model)
            else:
                raise ValueError(f"Model {model} not recognized.")
            return tokenizer
        
        def initializa_bert_transform(net, max_token_length=512):
            tokenizer = getBertTokenizer(net)
            def transform(item):
                if isinstance(item, tuple):
                    text = item[0]
                else:
                    text = item
                
                if not isinstance(text, str):
                    text = str(text)

                tokens = tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=max_token_length,
                    return_tensors='pt'
                )
                if net == 'bert-base-uncased':
                    x = torch.stack(
                        (tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids']),
                        dim=2
                    )
                elif net == 'distilbert-base-uncased':
                    x = torch.stack(
                        (tokens['input_ids'], tokens['attention_mask']),
                        dim=2
                    )
                x = torch.squeeze(x, dim=0)
                return x
            return transform
        
        transform_train = initializa_bert_transform(net, max_token_length)
        transform_test = transform_train
    elif data in ["DomainNet"]:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"No such dataset {data}")
    
    return transform_train, transform_test
    
def get_cifar10(batch_size, data_dir, train, val, test):
    cifar_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur", "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate"]
    severities = [1, 2, 3, 4 ,5]
    transform_train, transform_test = get_transforms(data='CIFAR10')    # ALL the same
    CIFAR10v1_path = os.path.join(data_dir, 'CIFAR10')
    train_set, train_loader, val_set, val_loader = None, None, None, None
    if train:
        train_set = torchvision.datasets.CIFAR10(root=CIFAR10v1_path, train=True, transform=transform_train, download=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if val:
        val_set = torchvision.datasets.CIFAR10(root=CIFAR10v1_path, train=True, transform=transform_train, download=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_sets, test_loaders, test_types = [], [], []
    if test:
        test_setv1 = torchvision.datasets.CIFAR10(root=CIFAR10v1_path, train=False, transform=transform_test, download=False)
        test_loaderv1 = torch.utils.data.DataLoader(test_setv1, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("v1_test")
        CIFAR10v2_path = os.path.join(data_dir, 'CIFAR10V2')
        test_setv2 = CIFAR10v2(root=CIFAR10v2_path, train=True, download=False, transform=transform_test)
        test_loaderv2 = torch.utils.data.DataLoader(test_setv2, batch_size, shuffle=False, num_workers=8)
        test_types.append("v2")

        test_sets.append(test_setv1)
        test_sets.append(test_setv2)
        test_loaders.append(test_loaderv1)
        test_loaders.append(test_loaderv2)

        CIFAR10_C_path = os.path.join(data_dir, 'CIFAR-10-C')
        for data in cifar_c:
            for severity in severities:
                test_set = CIFAR10_C(root=CIFAR10_C_path, data_type=data, severity=severity, transform=transform_test)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
                test_sets.append(test_set)
                test_loaders.append(test_loader)
                test_types.append(f"corruption_{data}_severity_{severity}")

    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types

def get_cifar100(batch_size, data_dir, train, val, test):
    cifar_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur",\
				 "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate",\
				  "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate" ]
    severities = [1, 2, 3, 4 ,5]

    transform_train, transform_test = get_transforms(data="CIFAR100")
    CIFAR100v1_path = os.path.join(data_dir, 'CIFAR100')
    train_set, train_loader, val_set, val_loader = None, None, None, None
    if train:
        train_set = torchvision.datasets.CIFAR100(root=CIFAR100v1_path, train=True, transform=transform_train, download=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
    if val:
        val_set = torchvision.datasets.CIFAR100(root=CIFAR100v1_path, train=True, transform=transform_train, download=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=8)
    test_sets, test_loaders, test_types = [], [], []
    if test:
        test_set1 = torchvision.datasets.CIFAR100(CIFAR100v1_path, train=False, transform=transform_test, download=False)
        test_loader1 = torch.utils.data.DataLoader(test_set1, batch_size, shuffle=False, num_workers=8)
        test_sets.append(test_set1)
        test_loaders.append(test_loader1)
        test_types.append("v1_test")
        CIFAR100_C_path = os.path.join(data_dir, 'CIFAR-100-C')
        for data in cifar_c:
            for severity in severities:
                test_set = CIFAR100_C(root=CIFAR100_C_path, data_type=data, severity=severity, transform=transform_test, download=False)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
                test_sets.append(test_set)
                test_loaders.append(test_loader)
                test_types.append(f"corruption_{data}_severity_{severity}")
    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types

def get_mnist(batch_size, data_dir, train, val, test):
    transform_train, transform_test = get_transforms(data = "MNIST")
    gray_transform_train, gray_transform_test = get_transforms(data="SVHN")
    
    MNIST_path = os.path.join(data_dir, 'MNIST')
    train_set, train_loader, val_set, val_loader = None, None, None, None
    if train:
        train_set = torchvision.datasets.MNIST(MNIST_path, train=True, download=False, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if val:
        val_set = torchvision.datasets.MNIST(MNIST_path, train=False, transform=transform_test, download=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

    test_sets, test_loaders, test_types = [], [], []
    if test:
        SVHN_path = os.path.join(data_dir, 'MNIST', 'SVHN')
        test_set2_1 = torchvision.datasets.SVHN(SVHN_path, split='train', download=False, transform=gray_transform_train)
        test_set2_2 = torchvision.datasets.SVHN(SVHN_path, split='test', download=False, transform=gray_transform_test)
        test_set2 = torch.utils.data.ConcatDataset([test_set2_1, test_set2_2])
        test_loader2 = torch.utils.data.DataLoader(test_set2, batch_size=batch_size, shuffle=False, num_workers=8)
        test_sets.append(test_set2)
        test_loaders.append(test_loader2)
        test_types.append("SVHN")

        USPS_path = os.path.join(data_dir, 'MNIST')
        test_set3_1 = USPS(root=USPS_path, train=True, transform=transform_train, download=False)
        test_set3_2 = USPS(root=USPS_path, train=False, transform=transform_test, download=False)
        test_set3 = torch.utils.data.ConcatDataset([test_set3_1, test_set3_2])
        test_loader3 = torch.utils.data.DataLoader(test_set3, batch_size=batch_size, shuffle=False, num_workers=8)
        test_sets.append(test_set3)
        test_loaders.append(test_loader3)
        test_types.append("USPS")

        QMNIST_path = os.path.join(data_dir, 'MNIST')
        test_set4 = torchvision.datasets.QMNIST(QMNIST_path, train=False, download=False, transform=transform_test)
        test_loader4 = torch.utils.data.DataLoader(test_set4, batch_size=batch_size, shuffle=False, num_workers=8)
        test_sets.append(test_set4)
        test_loaders.append(test_loader4)
        test_types.append("QMNIST")

    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types

def get_ImageNet(batch_size, data_dir, train=False, val=True, test=False):
    imagenet_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur", "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate" ]
    severities = [1, 2, 3, 4 ,5]

    transform_train, transform_test = get_transforms("ImageNet")
    ImageNet_v1_path = os.path.join(data_dir, 'imagenetv1')
    train_set, train_loader, val_set, val_loader = None, None, None, None
    if train:
        train_path = os.path.join(ImageNet_v1_path, 'train')
        train_set = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if val:
        val_path = os.path.join(ImageNet_v1_path, 'val')
        val_set = torchvision.datasets.ImageFolder(val_path, transform = transform_train)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    test_sets, test_loaders, test_types = [], [], []
    if test:
        test2_path = os.path.join(data_dir, 'imagenetv2', 'imagenetv2-matched-frequency-format-val')
        testsetv2 = torchvision.datasets.ImageFolder(test2_path, transform = transform_test)
        testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("v2_matched_frequency")
        test3_path = os.path.join(data_dir, 'imagenetv2', 'imagenetv2-threshold0.7-format-val')
        testsetv2_1 = torchvision.datasets.ImageFolder(test3_path, transform = transform_test)
        testloaderv2_1 = torch.utils.data.DataLoader(testsetv2_1, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("v2_threshold0_7")
        test4_path = os.path.join(data_dir, 'imagenetv2', 'imagenetv2-top-images-format-val')
        testsetv2_2 = torchvision.datasets.ImageFolder(test4_path, transform = transform_test)
        testloaderv2_2 = torch.utils.data.DataLoader(testsetv2_2, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("v2_top_images")
        test5_path = os.path.join(data_dir, 'imagenet-sketch')
        testsetv3 = torchvision.datasets.ImageFolder(test5_path, transform = transform_test)
        testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("sketch")

        test_sets.append(testsetv2)
        test_sets.append(testsetv2_1)
        test_sets.append(testsetv2_2)
        test_sets.append(testsetv3)

        test_loaders.append(testloaderv2)
        test_loaders.append(testloaderv2_1)
        test_loaders.append(testloaderv2_2)
        test_loaders.append(testloaderv3)

        for data in imagenet_c:
            for severity in severities:
                c_path = os.path.join(data_dir, 'imagenet-c', data, str(severity))
                test_set = torchvision.datasets.ImageFolder(c_path, transform=transform_test)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
                test_sets.append(test_set)
                test_loaders.append(test_loader)
                test_types.append(f"corruption_{data}_severity_{severity}")
    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types

def get_TinyImageNet(batch_size, data_dir, train=False, val=True, test=False):
    imagenet_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur",\
				 "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate",\
				  "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate" ]
    severities = [1, 2, 3, 4 ,5]

    transform_train, transform_test = get_transforms(data="Tiny_ImageNet")
    train_set, train_loader, val_set, val_loader = None, None, None, None

    Tiny_ImageNet_path = os.path.join(data_dir, 'TinyImagenet')
    if train:
        train_path = os.path.join(Tiny_ImageNet_path, 'imagenetv1', 'train')
        train_set = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if val:
        val_path = os.path.join(Tiny_ImageNet_path, 'imagenetv1', 'val')
        val_set = torchvision.datasets.ImageFolder(val_path, transform=transform_train)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    test_sets, test_loaders, test_types = [], [], []
    if test:
        test2_1_path = os.path.join(Tiny_ImageNet_path, 'imagenetv2', 'imagenetv2-matched-frequency-format-val')
        test_set2_1 = torchvision.datasets.ImageFolder(test2_1_path, transform=transform_test)
        test_loader2_1 = torch.utils.data.DataLoader(test_set2_1, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("v2_matched_frequency")
        test2_2_path = os.path.join(Tiny_ImageNet_path, 'imagenetv2', 'imagenetv2-threshold0.7-format-val')
        test_set2_2 = torchvision.datasets.ImageFolder(test2_2_path, transform=transform_test)
        test_loader2_2 = torch.utils.data.DataLoader(test_set2_2, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("v2_threshold0_7")
        test2_3_path = os.path.join(Tiny_ImageNet_path, 'imagenetv2', 'imagenetv2-top-images-format-val')
        test_set2_3 = torchvision.datasets.ImageFolder(test2_3_path, transform=transform_test)
        test_loader2_3 = torch.utils.data.DataLoader(test_set2_3, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("v2_top_images")
        test3_path = os.path.join(Tiny_ImageNet_path, 'imagenet-sketch')
        test_set3 = torchvision.datasets.ImageFolder(test3_path, transform=transform_test)
        test_loader3 = torch.utils.data.DataLoader(test_set3, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("sketch")
        test4_path = os.path.join(Tiny_ImageNet_path, 'imagenet-r')
        test_set4 = torchvision.datasets.ImageFolder(test4_path, transform=transform_test)
        test_loader4 = torch.utils.data.DataLoader(test_set4, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("reality")
        
        test_sets.append(test_set2_1)
        test_sets.append(test_set2_2)
        test_sets.append(test_set2_3)
        test_sets.append(test_set3)
        test_sets.append(test_set4)

        test_loaders.append(test_loader2_1)
        test_loaders.append(test_loader2_2)
        test_loaders.append(test_loader2_3)
        test_loaders.append(test_loader3)
        test_loaders.append(test_loader4)

        for data in imagenet_c:
            for severity in severities:
                c_path = os.path.join(Tiny_ImageNet_path, 'imagenet-c', data, str(severity))
                test_set = torchvision.datasets.ImageFolder(c_path, transform=transform_test)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
                test_sets.append(test_set)
                test_loaders.append(test_loader)
                test_types.append(f"corruption_{data}_severity_{severity}")
    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types

def get_imagenet_breeds(batch_size, data_dir, name=None, train=False, val=False, test=True):
    from robustness.tools.helpers import get_label_mapping
    from robustness.tools import folder
    from robustness.tools.breeds_helpers import make_entity13, make_entity30, make_living17, make_nonliving26

    if name == "living17":
        ret = make_living17(os.path.join(data_dir, 'imagenet_class_hierarchy'), split="good")
    elif name == "entity13":
        ret = make_entity13(os.path.join(data_dir, 'imagenet_class_hierarchy'), split="good")
    elif name == "entity30":
        ret = make_entity30(os.path.join(data_dir, 'imagenet_class_hierarchy'), split="good")
    elif name == "nonliving26":
        ret = make_nonliving26(os.path.join(data_dir, 'imagenet_class_hierarchy'), split="good")

    source_label_mapping = get_label_mapping('custom_imagenet', ret[1][0])
    target_label_mapping = get_label_mapping('custom_imagenet', ret[1][1])
    label_classNames = ret[2]

    transform_train, transform_test = get_transforms('ImageNet_Breeds')

    train_set, train_loader, val_set, val_loader = None, None, None, None
    if train:
        train_path = os.path.join(data_dir, 'imagenetv1', 'train')
        train_set = folder.ImageFolder(root=train_path, transform=transform_train, label_mapping=source_label_mapping)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if val:
        val_path = os.path.join(data_dir, 'imagenetv1', 'val')
        val_set = folder.ImageFolder(root=val_path, transform=transform_train, label_mapping=source_label_mapping)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

    imagenet_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur", "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate" ]
    severities = [1, 2, 3, 4 ,5]
    
    test_sets, test_loaders, test_types = [], [], []
    if test:
        test2_path_ = os.path.join(data_dir, 'imagenetv2', 'imagenetv2-matched-frequency-format-val')
        test_set2_ = folder.ImageFolder(test2_path_, transform = transform_test, label_mapping = source_label_mapping)
        test_loader2_ = torch.utils.data.DataLoader(test_set2_, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("same_v2_matched_frequency")

        test2_1_path_ = os.path.join(data_dir, 'imagenetv2', 'imagenetv2-threshold0.7-format-val')
        test_set2_1_ = folder.ImageFolder(test2_1_path_, transform = transform_test, label_mapping = source_label_mapping)
        test_loader2_1_ = torch.utils.data.DataLoader(test_set2_1_, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("same_v2_threshold0_7")

        test2_2_path_ = os.path.join(data_dir, 'imagenetv2', 'imagenetv2-top-images-format-val')
        test_set2_2_ = folder.ImageFolder(test2_2_path_, transform = transform_test, label_mapping = source_label_mapping)
        test_loader2_2_ = torch.utils.data.DataLoader(test_set2_2_, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("same_v2_top_images")

        test_sets.append(test_set2_)
        test_sets.append(test_set2_1_)
        test_sets.append(test_set2_2_)
        test_loaders.append(test_loader2_)
        test_loaders.append(test_loader2_1_)
        test_loaders.append(test_loader2_2_)

        for data in imagenet_c:
            for severity in severities:
                c_path_ = os.path.join(data_dir, 'imagenet-c', data, str(severity))
                test_set_ = folder.ImageFolder(c_path_, transform=transform_test, label_mapping=source_label_mapping)
                test_loader_ = torch.utils.data.DataLoader(test_set_, batch_size=batch_size, shuffle=False, num_workers=8)
                test_sets.append(test_set_)
                test_loaders.append(test_loader_)
                test_types.append(f"same_corruption_{data}_severity_{severity}")

        test2_path = os.path.join(data_dir, 'imagenetv2', 'imagenetv2-matched-frequency-format-val')
        test_set2 = folder.ImageFolder(test2_path, transform = transform_test, label_mapping = target_label_mapping)
        test_loader2 = torch.utils.data.DataLoader(test_set2, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("novel_v2_matched_frequency")

        test2_1_path = os.path.join(data_dir, 'imagenetv2', 'imagenetv2-threshold0.7-format-val')
        test_set2_1 = folder.ImageFolder(test2_1_path, transform = transform_test, label_mapping = target_label_mapping)
        test_loader2_1 = torch.utils.data.DataLoader(test_set2_1, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("novel_v2_threshold0_7")

        test2_2_path = os.path.join(data_dir, 'imagenetv2', 'imagenetv2-top-images-format-val')
        test_set2_2 = folder.ImageFolder(test2_2_path, transform = transform_test, label_mapping = target_label_mapping)
        test_loader2_2 = torch.utils.data.DataLoader(test_set2_2, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("novel_v2_top_images")

        test_sets.append(test_set2)
        test_sets.append(test_set2_1)
        test_sets.append(test_set2_2)
        test_loaders.append(test_loader2)
        test_loaders.append(test_loader2_1)
        test_loaders.append(test_loader2_2)

        for data in imagenet_c:
            for severity in severities:
                c_path = os.path.join(data_dir, 'imagenet-c', data, str(severity))
                test_set = folder.ImageFolder(c_path, transform=transform_test, label_mapping=target_label_mapping)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
                test_sets.append(test_set)
                test_loaders.append(test_loader)
                test_types.append(f"novel_corruption_{data}_severity_{severity}")
    
    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types, label_classNames

def get_camelyon17(batch_size, data_dir, train, val, test):
    data_set = Camelyon17Dataset(download=True, root_dir=data_dir)
    transform_train, transform_test = get_transforms("Camelyon17")
    if train:
        train_set = data_set.get_subset('train', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if val:
        val_set = data_set.get_subset('id_val', transform=transform_train)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    test_sets, test_loaders, test_types = [], [], []
    if test:
        test_set2 = data_set.get_subset('val', transform=transform_test)
        test_loader2 = torch.utils.data.DataLoader(test_set2, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("ood_val")
        test_set3 = data_set.get_subset('test', transform=transform_test)
        test_loader3 = torch.utils.data.DataLoader(test_set3, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("ood_test")
        
        test_sets.append(test_set2)
        test_sets.append(test_set3)
        test_loaders.append(test_loader2)
        test_loaders.append(test_loader3)

    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types

def get_fmow(batch_size, data_dir, train, val, test):
    data_set = FMoWDataset(download=False, root_dir=data_dir, use_ood_val=True)
    transform_train, transform_test = get_transforms("Fmow")
    train_set, train_loader, val_set, val_loader = None, None, None, None

    if train:
        train_set = data_set.get_subset('train', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if val:
        val_set = data_set.get_subset('id_val', transform=transform_train)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    test_sets, test_loaders, test_types = [], [], []
    if test:
        test_set2 = data_set.get_subset('val', transform=transform_test)
        test_set3 = data_set.get_subset('test', transform=transform_test)
        
        test_sets.append(test_set2)
        test_sets.append(test_set3)
        test_types.append("ood_val")
        test_types.append("ood_test")

        def get_groups(test_set):
            groups = data_set._eval_groupers['region'].metadata_to_group(test_set.metadata_array)
            group_testset = []
            for i in range(5):
                idx = np.where(groups==i)[0]
                group_testset.append(WILDSSubset(test_set, idx, None))
            return group_testset

        test_set2_groups = get_groups(test_set2)
        test_set3_groups = get_groups(test_set3)
        test_sets.extend(test_set2_groups)
        test_sets.extend(test_set3_groups)

        for i in range(5):
            test_types.append(f"ood_val_group{i}") 
        for i in range(5):
            test_types.append(f"ood_test_group{i}")

        for set in test_sets:
            test_loaders.append(torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=False, num_workers=8))

    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types

def get_rxrx1(batch_size, data_dir, train, val, test):
    dataset = RxRx1Dataset(download=False, root_dir=data_dir)
    transform_train, transform_test = get_transforms("Rxrx1")
    train_set, train_loader, val_set, val_loader = None, None, None, None
    if train:
        train_set = dataset.get_subset('train', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if val:
        val_set = dataset.get_subset('id_test', transform=transform_train)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    test_sets, test_loaders, test_types = [], [], []
    if test:
        test_set2 = dataset.get_subset('val', transform=transform_test)
        test_loader2 = torch.utils.data.DataLoader(test_set2, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("ood_val")
        test_set3 = dataset.get_subset('test', transform=transform_test)
        test_loader3 = torch.utils.data.DataLoader(test_set3, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("ood_test")

        test_sets.append(test_set2)
        test_sets.append(test_set3)
        test_loaders.append(test_loader2)
        test_loaders.append(test_loader3)

    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types

def get_amazon(batch_size, data_dir, net, max_token_length, train, val, test):
    dataset = AmazonDataset(download=False, root_dir=data_dir)
    transform_train, transform_test = get_transforms(data="Amazon", net=net, max_token_length=max_token_length)
    train_set, train_loader, val_set, val_loader = None, None, None, None
    if train:
        train_set = dataset.get_subset('train', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if val:
        val_set = dataset.get_subset('id_val', transform=transform_train)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    test_sets, test_loaders, test_types = [], [], []
    if test:
        test_set2 = dataset.get_subset('val', transform=transform_test)
        test_loader2 = torch.utils.data.DataLoader(test_set2, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("ood_val")
        test_set3 = dataset.get_subset('test', transform=transform_test)
        test_loader3 = torch.utils.data.DataLoader(test_set3, batch_size=batch_size, shuffle=False, num_workers=8)
        test_types.append("ood_test")
        test_sets.append(test_set2)
        test_sets.append(test_set3)
        test_loaders.append(test_loader2)
        test_loaders.append(test_loader3)

    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types

def get_civilcomments(batch_size, data_dir, net, max_token_length, train, val, test):
    dataset = CivilCommentsDataset(download=False, root_dir=data_dir)

    transform_train, transform_test = get_transforms(data="CivilComments", net=net, max_token_length=max_token_length)
    train_set, train_loader, val_set, val_loader = None, None, None, None
    if train:
        train_set = dataset.get_subset('train', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if val:
        val_set = dataset.get_subset('val', transform=transform_test)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    test_sets, test_loaders, test_types = [], [], []
    if test:
        test_set3 = dataset.get_subset('test', transform=transform_test)
        test_sets.append(test_set3)
        test_types.append("test")

        def get_groups(test_set):
            group_testset = []
            for i in range(7):
                groups = dataset._eval_groupers[i].metadata_to_group(test_set.metadata_array)

                idx = np.where(groups==1)[0]
                group_testset.append(WILDSSubset(test_set, idx, transform=None))

                idx = np.where(groups==3)[0]
                group_testset.append(WILDSSubset(test_set, idx, transform=None))
            return group_testset
        
        test_set3_groups = get_groups(test_set3)

        test_sets.extend(test_set3_groups)

        for i in range(7):
            test_types.append(f"test_grouper_{i}_group_1")
            test_types.append(f"test_grouper_{i}_group_3")

        for set in test_sets:
            test_loaders.append(torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=False, num_workers=8))

    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types

def get_domainnet(batch_size, data_dir, train=False, val=False, test=True):
    train_domain, val_domain = 'real', 'real'
    test_domains = ['clipart', 'painting', 'sketch']
    transform_train, transform_test = get_transforms(data='DomainNet')
    domain_path = os.path.join(data_dir, 'DomainNet')
    train_set, train_loader, val_set, val_loader = None, None, None, None

    if train:
        train_path = os.path.join(domain_path, 'train', train_domain)
        train_set = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if val:
        val_path = os.path.join(domain_path, 'test', val_domain)
        val_set = torchvision.datasets.ImageFolder(val_path, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

    test_sets, test_loaders, test_types = [], [], []
    if test:
        for domain in test_domains:
            cur_path = os.path.join(domain_path, domain)
            test_set = torchvision.datasets.ImageFolder(cur_path, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
            test_sets.append(test_set)
            test_loaders.append(test_loader)

            test_types.append(f"domain_{domain}")
    return train_set, train_loader, val_set, val_loader, test_sets, test_loaders, test_types



