from __future__ import absolute_import
import warnings

from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .msmt17 import MSMT17
from .cuhk_sysu import CUHK_SYSU
from .viper import VIPeR
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'msmt17': MSMT17,
    'cuhk_sysu': CUHK_SYSU,
    'viper': VIPeR,
}

import os.path as osp
from utils.data.preprocessor import Preprocessor
from utils.data import transforms as T
from torch.utils.data import DataLoader
from utils.data import IterLoader
from utils.data.sampler import RandomMultipleGallerySampler


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)


def get_data(name, data_dir, height, width, batch_size, workers, num_instances):
    if name == "cuhk_sysu":
        root = osp.join(data_dir, "cuhksysu4reid")
    elif name == "msmt17":
        root = osp.join(data_dir, "MSMT17")
    elif name == "dukemtmc":
        root = osp.join(data_dir, "DukeMTMC-reID")
    else:
        root = osp.join(data_dir, name)

    dataset = create(name, root)
    train_set = sorted(dataset.train)
    num_classes = dataset.num_train_pids
    iters = int(len(train_set) / batch_size)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    sampler = RandomMultipleGallerySampler(train_set, num_instances)
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=False, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(dataset.query + dataset.gallery),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=test_transformer),
                             batch_size=128, num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)

    return dataset, num_classes, train_loader, test_loader, init_loader, sampler
