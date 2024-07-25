import json
import numpy as np
import os
import torch
import torchvision
from tqdm import tqdm
from os.path import join, dirname
from random import sample
from FLDataset import FLDataset
from torchvision import transforms
import random

dataset_domains = {
    'vlcs': ["CALTECH", "LABELME", "PASCAL", "SUN"],
    'pacs': ["art_painting", "cartoon", "photo", "sketch"],
    'officehome': ["Art", "Clipart", "Product", "RealWorld"],
    'domainnet': ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
}

def distribute_clients_categorical(x, p, clients=400, beta=0.1):

    unique, counts = torch.Tensor(x.targets.float()).unique(return_counts=True)

    # Generate offsets within classes
    offsets = np.cumsum(np.broadcast_to(counts[:, np.newaxis], p.shape) * p, axis=1).astype('uint64')

    # Generate offsets for each class in the indices
    inter_class_offsets = np.cumsum(counts) - counts
    
    # Generate absolute offsets in indices for each client
    offsets = offsets + np.broadcast_to(inter_class_offsets[:, np.newaxis], offsets.shape).astype('uint64')
    offsets = np.concatenate([offsets, np.cumsum(counts)[:, np.newaxis]], axis=1).astype('uint64')

    # Use the absolute offsets as slices into the indices
    indices = []
    n_classes_by_client = []
    index_source = torch.LongTensor(np.argsort(x.targets))
    for client in range(clients):
        to_concat = []
        for noncontig_offsets in offsets[:, client:client + 2]:
            to_concat.append(index_source[slice(*noncontig_offsets)])
        indices.append(torch.cat(to_concat))
        n_classes_by_client.append(sum(1 for x in to_concat if x.numel() > 0))

    n_indices = np.array([x.numel() for x in indices])
    
    return indices, n_indices, n_classes_by_client


def distribute_clients_dirichlet(train, test, clients=400, batch_size=32, beta=0.1, rng=None):
    '''Distribute a dataset according to a Dirichlet distribution.
    '''

    rng = np.random.default_rng(rng)
    unique = torch.Tensor(train.targets.float()).unique()

    # Generate Dirichlet samples
    alpha = np.ones(clients) * beta
    p = rng.dirichlet(alpha, size=len(unique))

    # Get indices for train and test sets
    train_idx, _, __ = distribute_clients_categorical(train, p, clients=clients, beta=beta)
    test_idx, _, __ = distribute_clients_categorical(test, p, clients=clients, beta=beta)
    return train_idx, test_idx


# Follow the non-iid way described in FedAvg paper
# ```
# where we first sort the data by digit label, divide it into 200 shards of size 300, and assign each of 100 clients 2 shards. This is a
# pathological non-IID partition of the data, as most clients will only have examples of two digits.
# ```
def distribute_clients_categorical_follow_fedavg_way(labels, clients=100, cls_per_client=2, no_shards=200, cls_order=None):
    sort_data_idx = torch.argsort(labels)

    shard_size = int(len(labels) / no_shards)
    shards_data_idx = torch.split(sort_data_idx, shard_size)
    shards_per_client = int(no_shards / clients)

    classes_size = len(torch.unique(labels))
    if cls_order is None:
        cls_order = []
        M = int((clients * cls_per_client) / classes_size)
        for i in range(M):
            tmp_list = list(range(classes_size))
            random.shuffle(tmp_list)
            cls_order = cls_order+tmp_list

    clients_shard_no = []
    label_count = {}
    for i in range(0, len(cls_order), cls_per_client):
        cls_arr = cls_order[i:i+cls_per_client]
        shard_no_cls = []
        for cls1 in cls_arr:
            if cls1 not in label_count.keys():
                label_count[cls1] = 0
            t = int(no_shards / (clients * cls_per_client) ) # how many shards per class per client
            for j in range(t):
                shard_no = cls1*int(no_shards/classes_size)+ label_count[cls1] + j
                shard_no_cls.append(shard_no)
            label_count[cls1] += t
#         print(cls_arr, shard_no_cls)
        clients_shard_no.append(shard_no_cls)

    result = []
    for client_idx in range(len(clients_shard_no)):
        shard_no_arr = clients_shard_no[client_idx]
        t = []
        for shard_no in shard_no_arr:
            t.append(shards_data_idx[shard_no])
        client_data_idx = torch.cat(t,dim=0)
        result.append(client_data_idx)

    return result, cls_order


def distribute_clients_noniid(train, test, clients=100):
    '''Distribute a dataset according to a Dirichlet distribution.
    '''
    # Get indices for train and test sets
    train_labels = torch.Tensor(train.targets).int() if isinstance(train.targets, list) else train.targets
    train_idx, cls_order = distribute_clients_categorical_follow_fedavg_way(train_labels, clients=clients, cls_order=None)
    
    test_labels = torch.Tensor(test.targets).int() if isinstance(test.targets, list) else test.targets
    test_idx, _ = distribute_clients_categorical_follow_fedavg_way(test_labels, clients=clients, cls_order=cls_order)
    return train_idx, test_idx


def distribute_iid(train, test, clients=400, samples_per_client=40, batch_size=32, rng=None):
    '''Distribute a dataset in an iid fashion, i.e. shuffle the data and then
    partition it.'''

    rng = np.random.default_rng(rng)

    train_idx = np.arange(len(train.targets))
    rng.shuffle(train_idx)
    if samples_per_client < 1:
        samples_per_client = int(len(train.targets) / clients)
    print(f"samples_per_client: {samples_per_client}")
    train_idx = train_idx[:clients*samples_per_client]
    train_idx = train_idx.reshape((clients, samples_per_client))

    test_idx = np.arange(len(test.targets))
    rng.shuffle(test_idx)
    test_samples_per_client = int(len(test.targets) / clients)
    test_idx = test_idx[:clients*test_samples_per_client]
    test_idx = test_idx.reshape((clients, test_samples_per_client))

    return train_idx, test_idx


def get_mnist_or_cifar10(dataset='mnist', mode='dirichlet', path=None, clients=400,
                         classes=2, samples=20, batch_size=32, beta=0.1,
                         unbalance_rate=1.0, rng=None, **kwargs):
    '''Sample a FL dataset from MNIST, as in the LotteryFL paper.

    Parameters:
    dataset : str
        either mnist or cifar10
    path : str
        currently unused
    clients : int
        number of clients among which the dataset should be distributed
    classes : int
        number of classes each client gets in its training set
    samples : int
        number of samples per class each client gets in its training set
    batch_size : int
        batch size to use for the DataLoaders
    unbalance_rate : float
        how different the number of samples in the second class can be
        from the first, e.g. specifying samples=20, unbalance_rate=0.5 means
        the second class can be anywhere from 10-40 samples.
    rng : numpy random generator
        the RNG to use to shuffle samples.
        if None, we will grab np.random.default_rng().
        

    Returns: dict of client_id -> 2-tuples of DataLoaders
    '''

    if dataset not in ('mnist', 'cifar10', 'cifar100'):
        raise ValueError(f'unsupported dataset {dataset}')

    if path is None:
        user_home_path = os.path.expanduser ('~')
        path = os.path.join(user_home_path, 'Datasets', dataset)

    rng = np.random.default_rng(rng)

    if mode in ('dirichlet', 'iid','noniid'):
        if dataset == 'mnist':
            xfrm = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
            train = torchvision.datasets.MNIST(path, train=True, download=True, transform=xfrm)
            test = torchvision.datasets.MNIST(path, train=False, download=True, transform=xfrm)

        elif dataset == 'cifar10':
            xfrm = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train = torchvision.datasets.CIFAR10(path, train=True, download=True, transform=xfrm)
            test = torchvision.datasets.CIFAR10(path, train=False, download=True, transform=xfrm)
        elif dataset == 'cifar100':
            xfrm = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train = torchvision.datasets.CIFAR100(path, train=True, download=True, transform=xfrm)
            test = torchvision.datasets.CIFAR100(path, train=False, download=True, transform=xfrm)

        if mode == 'dirichlet':
            train_idx, test_idx = distribute_clients_dirichlet(train, test, clients=clients, beta=beta)
        elif mode == 'iid':
            train_idx, test_idx = distribute_iid(train, test, clients=clients, samples_per_client=samples)
        elif mode =='noniid':
            train_idx, test_idx = distribute_clients_noniid(train, test, clients=clients)

    elif mode == 'lotteryfl':
        if dataset == 'mnist':
            from non_iid.dataset.mnist_noniid import get_dataset_mnist_extr_noniid as g
        elif dataset == 'cifar10':
            from non_iid.dataset.cifar10_noniid import get_dataset_cifar10_extr_noniid as g
        elif dataset == 'cifar100':
            from cifar100_noniid import get_dataset_cifar100_extr_noniid as g
        else:
            raise ValueError(f'dataset {dataset} is not supported by lotteryfl')

        train, test, train_idx, test_idx = g(clients, classes, samples, unbalance_rate)

    # Generate DataLoaders
    loaders = {}
    for i in range(clients):
        train_sampler = torch.LongTensor(train_idx[i])
        test_sampler = torch.LongTensor(test_idx[i])

        if len(train_sampler) == 0 or len(test_sampler) == 0:
            # ignore empty clients
            continue

        # shuffle
        train_sampler = rng.choice(train_sampler, size=train_sampler.shape, replace=False)
        test_sampler = rng.choice(test_sampler, size=test_sampler.shape, replace=False)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                                  sampler=test_sampler)
        loaders[i] = (0, train_loader, test_loader)

    return loaders


def get_mnist(*args, **kwargs):
    return get_mnist_or_cifar10('mnist', *args, **kwargs)


def get_cifar10(*args, **kwargs):
    return get_mnist_or_cifar10('cifar10', *args, **kwargs)


def get_cifar100(*args, **kwargs):
    return get_mnist_or_cifar10('cifar100', *args, **kwargs)

def get_pacs(*args, **kwargs):
    return get_pacs_officehome_vlcs('pacs')

def get_officehome(*args, **kwargs):
    return get_pacs_officehome_vlcs('officehome')

def get_vlcs(*args, **kwargs):
    return get_pacs_officehome_vlcs('vlcs')

def get_path_dataset_info(domain_name, split):
    postfix_pattern = '_' + split + '.txt'
    if domain_name in dataset_domains['officehome']:
            postfix_pattern = '.txt'
    dataset_path = join(dirname(__file__), 'data/txt_lists', ('%s'+postfix_pattern) % domain_name)
    sample_paths, labels = get_dataset_paths(dataset_path)
    return sample_paths, labels

def get_dataset_paths(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()
    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))
    return file_names, labels

def get_random_subset(names, labels, percent):
    needVal = True
    if not percent > 0.0:
        needVal = False
        percent = 0.1
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]

    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]

    if needVal:
        return name_train, name_val, labels_train, labels_val
    else:
        return names, name_val, labels, labels_val

def get_pacs_officehome_vlcs(dataset='pacs', batch_size=32, **kwargs):
    loaders = {}
    img_tr = [transforms.Resize((222, 222)), transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    img_transformer = transforms.Compose(img_tr)
    for dname in dataset_domains[dataset]: 
        paths_train, labels_train = get_path_dataset_info(dname, 'train')
        paths_test, labels_test = get_path_dataset_info(dname, 'test')
        # paths_train, paths_val, labels_train, labels_val = get_random_subset(paths, labels, 0.1)
        trainset = FLDataset(labels_train, sample_paths = paths_train,transformer=img_transformer)
        testset = FLDataset(labels_test, sample_paths=paths_test, transformer=img_transformer)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
        loaders[dname] = (0, train_loader, test_loader) # 0 here is the gpu device id.
    return loaders


def get_emnist(path='../leaf/data/femnist/data', min_samples=0, batch_size=32,
               val_size=0.2, **kwargs):
    '''Read the Federated EMNIST dataset, from the LEAF benchmark.
    The number of clients, classes per client, samples per class, and
    class imbalance are all provided as part of the dataset.

    Parameters:
    path : str
        dataset root directory
    batch_size : int
        batch size to use for DataLoaders
    val_size : float
        the relative proportion of test samples each client gets

    Returns: dict of client_id -> (train_loader, test_loader)
    '''

    EMNIST_SUBDIR = 'all_data'

    loaders = {}
    for fn in tqdm(os.listdir(os.path.join(path, EMNIST_SUBDIR))):
        fn = os.path.join(path, EMNIST_SUBDIR, fn)
        with open(fn) as f:
            subset = json.load(f)
        print(len(subset['users']))
        for uid in subset['users']:
            user_data = subset['user_data'][uid]
            data_x = (torch.FloatTensor(x).reshape((1, 28, 28)) for x in user_data['x'])
            data = list(zip(data_x, user_data['y']))

            # discard clients with less than min_samples of training data
            if len(data) < min_samples:
                continue

            n_train = int(len(data) * (1 - val_size))
            data_train = data[:n_train]
            data_test = data[n_train:]
            train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
            test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)

            loaders[uid] = (train_loader, test_loader)


    return loaders


def get_dataset(dataset, devices=['0'], **kwargs):
    '''Fetch the requested dataset, caching if needed

    Parameters:
    dataset : str
        either 'mnist' or 'emnist'
    devices : torch.device-like
        devices to cache the data on. If None, then minimal caching will be done.
    **kwargs
        passed to get_mnist or get_emnist

    Returns: dict of client_id -> (device, train_loader, test_loader)
    '''

    DATASET_LOADERS = {
        'mnist': get_mnist,
        'emnist': get_emnist,
        'cifar10': get_cifar10,
        'cifar100': get_cifar100,
        'pacs': get_pacs,
        'officehome': get_officehome,
        'vlcs': get_vlcs,
    }

    if dataset not in DATASET_LOADERS:
        raise ValueError(f'unknown dataset {dataset}. try one of {list(DATASET_LOADERS.keys())}')

    loaders = DATASET_LOADERS[dataset](**kwargs)

    # Cache the data on the given devices. (此处是为了多显卡训练，将数据分配到不同的显卡上)
    new_loaders = {}
    for i, (uid, loader) in enumerate(loaders.items()):
        device = devices[i % len(devices)]
        if len(loader) == 3:
            (_, train_loader, test_loader) = loader
            new_loaders[uid] = (device, train_loader, test_loader)
        else:
            (train_loader, test_loader) = loader
            train_data = [(x.to(device), y.to(device)) for x, y in train_loader]
            test_data = [(x.to(device), y.to(device)) for x, y in test_loader]
            new_loaders[uid] = (device, train_data, test_data)

    return new_loaders

