import ast
import io
import logging
import os
import pickle

import torch
from PIL import Image, ImageFile

import logging
import random
import traceback

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, pseudo_index=-1, skip_broken=False, new_index='next'):
        super(BaseDataset, self).__init__()
        self.pseudo_index = pseudo_index
        self.skip_broken = skip_broken
        self.new_index = new_index
        if new_index not in ('next', 'rand'):
            raise ValueError('new_index not one of ("next", "rand")')

    def __getitem__(self, index):
        # in some pytorch versions, input index will be torch.Tensor
        index = int(index)

        # if sampler produce pseudo_index,
        # randomly sample an index, and mark it as pseudo
        if index == self.pseudo_index:
            index = random.randrange(len(self))
            pseudo = 1
        else:
            pseudo = 0

        while True:
            try:
                sample = self.getitem(index)
                break
            except Exception as e:
                if self.skip_broken and not isinstance(e, NotImplementedError):
                    if self.new_index == 'next':
                        new_index = (index + 1) % len(self)
                    else:
                        new_index = random.randrange(len(self))
                    logging.warn(
                        'skip broken index [{}], use next index [{}]'.format(
                            index, new_index))
                    index = new_index
                else:
                    logging.error('index [{}] broken'.format(index))
                    traceback.print_exc()
                    logging.error(e)
                    raise e

        sample['index'] = index
        sample['pseudo'] = pseudo
        return sample

    def getitem(self, index):
        raise NotImplementedError


DATA_INFO = {
    'cifar10': {
        'num_classes': 10,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/train_cifar10.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cifar10.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10.txt'
            }
        },
        'csid': {
            'datasets': ['cifar10c'],
            'cinic10': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cinic10.txt'
            },
            'cifar10c': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10c.txt'
            }
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar100', 'tin'],
                'cifar100': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_cifar100.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_places365.txt'
                },
            }
        }
    },
    'cifar100': {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/train_cifar100.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_cifar100.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/test_cifar100.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar10', 'tin'],
                'cifar10': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_cifar10.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_places365.txt'
                }
            },
        }
    },
    'imagenet200': {
        'num_classes': 200,
        'id': {
            'train': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/train_imagenet200.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_imagenet200.txt'
            },
            'test': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_v2.txt'
            },
            'imagenet_c': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_c.txt'
            },
            'imagenet_r': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir':
                    'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_openimage_o.txt'
                },
            },
        }
    },
    'imagenet': {
        'num_classes': 1000,
        'id': {
            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/train_imagenet.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/val_imagenet.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/test_imagenet.txt'
            }
        },
        'csid': {
            'datasets':
            ['imagenet_v2', 'imagenet_c', 'imagenet_r', 'imagenet_es'],
            'imagenet_v2': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_v2.txt'
            },
            'imagenet_c': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_c.txt'
            },
            'imagenet_r': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_r.txt'
            },
            'imagenet_es': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_es.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/imagenet/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_openimage_o.txt'
                },
            },
        }
    },
}

import torchvision.transforms as tvs_trans

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True

oodb_class_num = {
    "patternnet": 19,
    "dtd": 23
}

for variant in range(3):
    for oodb_dataset_name in ["patternnet", "dtd"]:
        oodb_dataset_name_ood = "dtd" if oodb_dataset_name == "patternnet" else "patternnet"
        DATA_INFO[f"ooddb_{oodb_dataset_name}_{variant}"] = {
            'num_classes': oodb_class_num[oodb_dataset_name],
            'id': {
                'train': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': f"benchmark_imglist/ooddb/ooddb_{oodb_dataset_name}_id_{variant}_train.txt"
                },
                'val': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': f"benchmark_imglist/ooddb/ooddb_{oodb_dataset_name}_id_{variant}_test.txt"
                },
                'test': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': f"benchmark_imglist/ooddb/ooddb_{oodb_dataset_name}_id_{variant}_test.txt"
                }
            },
            'ood': {
                'near': {
                    f"ooddb_{oodb_dataset_name}_{variant}": {
                        'data_dir': 'images_largescale/',
                        'imglist_path': f"benchmark_imglist/ooddb/ooddb_{oodb_dataset_name}_ood_{variant}_test.txt"
                    },
                },
                'far': {
                    'cifar10': {
                        'data_dir': 'images_classic/',
                        'imglist_path':
                        'benchmark_imglist/cifar100/test_cifar10.txt'
                    },
                    'tin': {
                        'data_dir': 'images_classic/',
                        'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                    },
                    f"ooddb_{oodb_dataset_name_ood}_{variant}": {
                        'data_dir': 'images_largescale/',
                        'imglist_path': f"benchmark_imglist/ooddb/ooddb_{oodb_dataset_name_ood}_ood_{variant}_test.txt"
                    },
                    'mnist': {
                        'data_dir': 'images_classic/',
                        'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                    },
                    'svhn': {
                        'data_dir': 'images_classic/',
                        'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                    },
                    'texture': {
                        'data_dir': 'images_classic/',
                        'imglist_path':
                        'benchmark_imglist/cifar100/test_texture.txt'
                    },
                    'places365': {
                        'data_dir': 'images_classic/',
                        'imglist_path':
                        'benchmark_imglist/cifar100/test_places365.txt'
                    },
                }
            }
        }

class ImglistDataset(BaseDataset):
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(ImglistDataset, self).__init__(**kwargs)

        self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}

        try:
            with open(path, 'rb') as f:
                content = f.read()
            filebytes = content
            buff = io.BytesIO(filebytes)
            image = Image.open(buff).convert('RGB')
            sample['img'] = self.transform_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['gt_label'] = 0
            except AttributeError:
                sample['gt_label'] = int(extra_str)
            

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        #return sample['data'], sample['label']
        return sample


class ImglistDataset_CLIPWds(BaseDataset):
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 wds_index_json,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(ImglistDataset_CLIPWds, self).__init__(**kwargs)

        import wids

        self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen

        self.wds_index_json = wds_index_json
        self.wds_dataset = wids.ShardListDataset(self.wds_index_json, cache_dir=f'./cache/wids/{name}', keep=True)
        print(f"Loaded {self.wds_index_json} with {len(self.wds_dataset)} samples")

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}

        try:
            with open(path, 'rb') as f:
                content = f.read()
            filebytes = content
            buff = io.BytesIO(filebytes)
            image = Image.open(buff).convert('RGB')
            sample['data'] = self.transform_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)
            
            wds_sample = self.wds_dataset[index]
            sample["clip_feat"] = pickle.loads(wds_sample['.clip_feat.pyd'].read())
            #print(f"clip_feat.shape: {clip_feat.shape}")

        except Exception as e:
            print(self.wds_index_json, index, len(self.wds_dataset))
            logging.error('[{}] broken'.format(path))
            raise e
        return sample


def get_dataloader(dataset, loader_kwargs):
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)

def get_id_dataset(id_name, split, preprocessor, data_root='./data', args=None):
    data_info = DATA_INFO[id_name]
    totensor_tf = tvs_trans.ToTensor()

    if args is None or args.data_mode == "standard":
        return ImglistDataset(
        name='_'.join((id_name, split)),
        imglist_pth=os.path.join(data_root,
                                    data_info['id'][split]['imglist_path']),
        data_dir=os.path.join(data_root,
                                data_info['id'][split]['data_dir']),
        num_classes=data_info['num_classes'],
        preprocessor=preprocessor,
        data_aux_preprocessor=totensor_tf)
    elif args.data_mode == "webdataset":

        dataset_folder_name = {
            "ooddb_patternnet_0": "PatternNet",
            "ooddb_dtd_0": "DTD",
            "imagenet200": "imagenet200",
            "imagenet": "imagenet_1k"
        }

        wds_index_filename = f"{split}_clipfeat_{args.clip_architecture}-{args.clip_pretrained}-{args.aux_dim}_index.json"
        return ImglistDataset_CLIPWds(
        name='_'.join((id_name, split)),
        imglist_pth=os.path.join(data_root,
                                    data_info['id'][split]['imglist_path']),
        data_dir=os.path.join(data_root,
                                data_info['id'][split]['data_dir']),
        wds_index_json=os.path.join(data_root,data_info['id'][split]['data_dir'],dataset_folder_name.get(id_name,id_name),wds_index_filename), # TODO: fix data folder
        num_classes=data_info['num_classes'],
        preprocessor=preprocessor,
        data_aux_preprocessor=totensor_tf)

def get_ood_dataset(id_name, ood_name, preprocessor, split="near", data_root='./data'):
    data_info = DATA_INFO[id_name]
    print(data_info)
    totensor_tf = tvs_trans.ToTensor()
    
    return ImglistDataset(
    name='_'.join((id_name, ood_name)),
    imglist_pth=os.path.join(data_root,
                                data_info['ood'][split][ood_name]['imglist_path']),
    data_dir=os.path.join(data_root,
                            data_info['ood'][split][ood_name]['data_dir']),
    num_classes=data_info['num_classes'],
    preprocessor=preprocessor,
    data_aux_preprocessor=totensor_tf)

def get_ood_dict(id_name):
    if "cifar" in id_name:
        ood_datasets = {
            'svhn': 'far',
            'texture': 'far',
            'tin': 'near',
            'places365': 'far',
            'mnist': 'far',
            'cifar100' if id_name == 'cifar10' else 'cifar10': 'near'
        }
    elif "imagenet" in id_name:
        ood_datasets = {
            'inaturalist': 'far',
            'textures': 'far',
            'openimage_o': 'far',
            'ninco': 'near',
            'ssb_hard': 'near',
        }
    elif "ooddb" in id_name:
        ood_datasets = {
           id_name: 'near',
           'svhn': 'far',
            'ooddb_patternnet_0' if 'dtd' in id_name else 'ooddb_dtd_0': 'far',
            'tin': 'far',
            'places365': 'far',
            'mnist': 'far',
            'cifar10': 'far'
        }
    return ood_datasets

cifar100_labels = [
    'apple',  # id 0
    'aquarium_fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer_keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm',
]


# make a mapping from imagenet200 class idx to class name

# first, load the txt file containing path and label: data/benchmark_imglist/imagenet200/train_imagenet200.txt
# then, split the line by space and get the label

with open('../OpenOOD/data/benchmark_imglist/imagenet200/train_imagenet200.txt') as f:
    lines_imagenet200 = f.readlines()

imagenet200_label_to_wordnet_id = {}


for line in lines_imagenet200:
    path = line.split(' ')[0] # format imagenet_1k/train/n01443537/n01443537_10007.JPEG
    label = int(line.split(' ')[1].strip()) # format 0
    wordnet_id = path.split('/')[2] # format n01443537
    imagenet200_label_to_wordnet_id[label] = wordnet_id

imagenet1k_label_to_wordnet_id = {}

with open('../OpenOOD/data/benchmark_imglist/imagenet/train_imagenet.txt') as f:
    lines_imagenet1k = f.readlines()

for line in lines_imagenet1k:
    path = line.split(' ')[0] # format imagenet_1k/train/n01443537/n01443537_10007.JPEG
    label = int(line.split(' ')[1].strip()) # format 0
    wordnet_id = path.split('/')[2] # format n01443537
    imagenet1k_label_to_wordnet_id[label] = wordnet_id

#print(imagenet200_label_to_wordnet_id)

wordnet_id_to_class_name = {}
wordnet_id_to_class_idx = {}

# download https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
import json
import requests
import os

IMAGENET_CLASSNAMES = (
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
    "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
    "box turtle", "banded gecko", "green iguana", "Carolina anole",
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
    "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
    "American alligator", "triceratops", "worm snake", "ring-necked snake",
    "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
    "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
    "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
    "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
    "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
    "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
    "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
    "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
    "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
    "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
    "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
    "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
    "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
    "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
    "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
    "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
    "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
    "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
    "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
    "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
    "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
    "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
    "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
    "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
    "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
    "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
    "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
    "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
    "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
    "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
    "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
    "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
    "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
    "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
    "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
    "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
    "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
    "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
    "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
    "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
    "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
    "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
    "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
    "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
    "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
    "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
    "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
    "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
    "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
    "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
    "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
    "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
    "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
    "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
    "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
    "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
    "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
    "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
    "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
    "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
    "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
    "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
    "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
    "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
    "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
    "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
    "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
    "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
    "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
    "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
    "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
    "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
    "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
    "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
    "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
    "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
    "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
    "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
    "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
    "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
    "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
    "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
    "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
    "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
    "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
    "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
    "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
    "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
    "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
    "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
    "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
    "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
    "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
    "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
    "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
    "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
    "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
    "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
    "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
    "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
    "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
)

if not os.path.exists('imagenet_class_index.json'):
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    response = requests.get(url)
    with open('imagenet_class_index.json', 'wb') as f:
        f.write(response.content)

with open('imagenet_class_index.json') as f:
    class_index = json.load(f)

for key, value in class_index.items():
    wordnet_id = value[0]
    class_name = value[1]
    wordnet_id_to_class_name[wordnet_id] = class_name
    wordnet_id_to_class_idx[wordnet_id] = int(key)

#print(wordnet_id_to_class_name)

imagenet200_label_to_class_idx = {}
for label, wordnet_id in imagenet200_label_to_wordnet_id.items():
    class_idx = wordnet_id_to_class_idx[wordnet_id]
    imagenet200_label_to_class_idx[label] = class_idx

imagenet1k_label_to_class_idx = {}
for label, wordnet_id in imagenet1k_label_to_wordnet_id.items():
    class_idx = wordnet_id_to_class_idx[wordnet_id]
    imagenet1k_label_to_class_idx[label] = class_idx

imagenet200_label_to_class_name = {}
for orig_idx, in_idx in imagenet200_label_to_class_idx.items():
    imagenet200_label_to_class_name[orig_idx] = IMAGENET_CLASSNAMES[in_idx]

imagenet1k_label_to_class_name = {}
for orig_idx, in_idx in imagenet1k_label_to_class_idx.items():
    imagenet1k_label_to_class_name[orig_idx] = IMAGENET_CLASSNAMES[in_idx]
#print(imagenet200_label_to_class_name)

def get_label_to_class_mapping(id_name, ooddb_split="train",sanity_check=True):
    
    if id_name == "cifar100":
        mapping = dict(zip(range(len(cifar100_labels)),cifar100_labels))
        
    elif id_name == "imagenet200":
        mapping = imagenet200_label_to_class_name
    elif id_name == "imagenet":
        mapping = imagenet1k_label_to_class_name
    elif id_name == "ooddb_patternnet_0":
        from OODDB.utils import get_dataset_split_info
        _,_, class_idx_to_name = get_dataset_split_info(
            dataset="patternnet",
            split=ooddb_split,
            data_order=0,
        )
        mapping = class_idx_to_name
    elif id_name == "ooddb_dtd_0":
        from OODDB.utils import get_dataset_split_info
        _,_, class_idx_to_name = get_dataset_split_info(
            dataset="dtd",
            split=ooddb_split,
            data_order=0,
        )
        print(class_idx_to_name)
        mapping = class_idx_to_name

    mapping ={k:v.replace("_"," ") for k,v in mapping.items()}

    if sanity_check:
        assert len(mapping) == DATA_INFO[id_name]["num_classes"], \
        f"Number of classes in mapping ({len(mapping)}) does not match number of classes in dataset ({DATA_INFO[id_name]['num_classes']})"
    return mapping
    
if __name__ == "__main__":
    
    print("--patternnet")
    id_classes = get_label_to_class_mapping("ooddb_patternnet_0", ooddb_split="train")
    all_classes = get_label_to_class_mapping("ooddb_patternnet_0", ooddb_split="test", sanity_check=False)

    ood_classes = {k:v for k,v in all_classes.items() if k not in id_classes}

    print(f"ID classes: {id_classes}")
    print(f"OOD classes: {ood_classes}")

    print("-dtd")
    id_classes = get_label_to_class_mapping("ooddb_dtd_0", ooddb_split="train")
    all_classes = get_label_to_class_mapping("ooddb_dtd_0", ooddb_split="test", sanity_check=False)

    ood_classes = {k:v for k,v in all_classes.items() if k not in id_classes}

    print(f"ID classes: {id_classes}")
    print(f"OOD classes: {ood_classes}")

    for id_name in ["cifar100","imagenet200","imagenet"]:
        print(f"------------",id_name)
        id_classes = get_label_to_class_mapping(id_name)
        print(id_classes)