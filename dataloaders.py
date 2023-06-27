import os
from tempfile import TemporaryFile
from unittest import TestLoader
from torch.utils.data.dataset import Subset
from types import SimpleNamespace
import numpy as np


from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.dataset_phc import PHC
from data.dataset_phc_style import PHC_style

# Gone
# Import own files
from metadata_manager import *

# TODO: commented out because cv2 problems
from data.dataset_isic_style import *
from data.dataset_isic import *
from data.dataset_phc_style import *
from data.dataset_phc import *


def get_dataloader(
    task, split, batch_size, shuffle=True, splitratio=[0.6, 0.2, 0.2], randomsplit=False
):
    """
    Returns dataloader for training/validation/testing

    Args
        task: (string) which dataset/task for constructing dataloader
        split: (string) train/val/test
        batch_size: (int) batch size
        shuffle: (Bool) data randomly ordered?
    """

    meta_dict = get_meta(task)
    meta = SimpleNamespace(**meta_dict)

    if task == "isic3":
        # path_parent = os.path.dirname(os.getcwd())
        # data_path = os.path.join(path_parent, meta.all_data_path)
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = ISIC(trafo, apply_symmetric_transforms=False, data_path=data_path)
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                ISIC(trafo, apply_symmetric_transforms=True, data_path=data_path),
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                ISIC(trafo, apply_symmetric_transforms=False, data_path=data_path),
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                ISIC(trafo, apply_symmetric_transforms=False, data_path=data_path),
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    if task == "isic3_style":
        # path_parent = os.path.dirname(os.getcwd())
        # data_path = os.path.join(path_parent, meta.all_data_path)
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = ISIC_style(
            trafo, apply_symmetric_transforms=False, data_path=data_path
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                ISIC_style(trafo, apply_symmetric_transforms=True, data_path=data_path),
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                ISIC_style(
                    trafo, apply_symmetric_transforms=False, data_path=data_path
                ),
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                ISIC_style(
                    trafo, apply_symmetric_transforms=False, data_path=data_path
                ),
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    if task == "isic3_style_0":
        # path_parent = os.path.dirname(os.getcwd())
        # data_path = os.path.join(path_parent, meta.all_data_path)
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = ISIC_style_subset(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=0
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                ISIC_style_subset(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=0
                ),
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                ),
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                ),
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    if task == "isic3_style_1":
        # path_parent = os.path.dirname(os.getcwd())
        # data_path = os.path.join(path_parent, meta.all_data_path)
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = ISIC_style_subset(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=1
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                ISIC_style_subset(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=1
                ),
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                ),
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                ),
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    if task == "isic3_style_2":
        # path_parent = os.path.dirname(os.getcwd())
        # data_path = os.path.join(path_parent, meta.all_data_path)
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = ISIC_style_subset(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=2
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                ISIC_style_subset(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=2
                ),
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=2,
                ),
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=2,
                ),
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    if task == "isic3_style_concat":
        # this concatenates the train splits from isic3_style0, isic3_style1, isic3_style2 so that the style probabilistic U-Net does not see the test data during training

        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])

        # Build dataset style 0 only
        dataset = ISIC_style_subset(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=0
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset_style0_train = Subset(
                ISIC_style_subset(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=0
                ),
                train_indices,
            )

        if split == "val":
            dataset_style0_val = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                ),
                val_indices,
            )

        if split == "test":
            dataset_style0_test = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                ),
                test_indices,
            )

        # Build dataset style 1 only
        dataset = ISIC_style_subset(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=1
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset_style1_train = Subset(
                ISIC_style_subset(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=1
                ),
                train_indices,
            )

        if split == "val":
            dataset_style1_val = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                ),
                val_indices,
            )

        if split == "test":
            dataset_style1_test = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                ),
                test_indices,
            )

        # Build dataset style 2 only
        dataset = ISIC_style_subset(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=2
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset_style2_train = Subset(
                ISIC_style_subset(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=2
                ),
                train_indices,
            )

        if split == "val":
            dataset_style2_val = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=2,
                ),
                val_indices,
            )

        if split == "test":
            dataset_style2_test = Subset(
                ISIC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=2,
                ),
                test_indices,
            )

        # Concatenate the datasets
        if split == "train":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_train, dataset_style1_train, dataset_style2_train]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_val, dataset_style1_val, dataset_style2_val]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_test, dataset_style1_test, dataset_style2_test]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    if task == "isic3_dynamic_aug":
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = ISIC_dynamic_augmentation(
            trafo, apply_symmetric_transforms=False, data_path=data_path
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                ISIC_dynamic_augmentation(
                    trafo, apply_symmetric_transforms=True, data_path=data_path
                ),
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                ISIC_dynamic_augmentation(
                    trafo, apply_symmetric_transforms=False, data_path=data_path
                ),
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                ISIC_dynamic_augmentation(
                    trafo, apply_symmetric_transforms=False, data_path=data_path
                ),
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    if task == "phc":
        # path_parent = os.path.dirname(os.getcwd())
        # data_path = os.path.join(path_parent, meta.all_data_path)
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = PHC(trafo, apply_symmetric_transforms=False, data_path=data_path)

        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                PHC(trafo, apply_symmetric_transforms=True, data_path=data_path),
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                PHC(trafo, apply_symmetric_transforms=False, data_path=data_path),
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                PHC(trafo, apply_symmetric_transforms=False, data_path=data_path),
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    if task == "phc_style":
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = PHC_style(
            trafo, apply_symmetric_transforms=False, data_path=data_path
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                PHC_style(trafo, apply_symmetric_transforms=True, data_path=data_path),
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                PHC_style(trafo, apply_symmetric_transforms=False, data_path=data_path),
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                PHC_style(trafo, apply_symmetric_transforms=False, data_path=data_path),
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    if task == "phc_style_0":
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = PHC_style_subset(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=0
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                PHC_style_subset(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=0
                ),
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                PHC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                ),
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                PHC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                ),
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    if task == "phc_style_1":
        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])
        dataset = PHC_style_subset(
            trafo, apply_symmetric_transforms=False, data_path=data_path, style=1
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset = Subset(
                PHC_style_subset(
                    trafo, apply_symmetric_transforms=True, data_path=data_path, style=1
                ),
                train_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = Subset(
                PHC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                ),
                val_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = Subset(
                PHC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                ),
                test_indices,
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    if task == "phc_style_concat":
        # this concatenates the train splits from phc_style0, phc_style1 so that the style probabilistic U-Net does not see the test data during training

        data_path = meta.all_data_path
        trafo = transforms.Compose([transforms.ToTensor()])

        # Build dataset style 0 only
        dataset = PHC_style_subset(
            trafo,
            apply_symmetric_transforms=False,
            data_path=data_path,
            style=0,
            return_all_annotations=True,
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset_style0_train = Subset(
                PHC_style_subset(
                    trafo,
                    apply_symmetric_transforms=True,
                    data_path=data_path,
                    style=0,
                    return_all_annotations=True,
                ),
                train_indices,
            )

        if split == "val":
            dataset_style0_val = Subset(
                PHC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                    return_all_annotations=True,
                ),
                val_indices,
            )

        if split == "test":
            dataset_style0_test = Subset(
                PHC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=0,
                    return_all_annotations=True,
                ),
                test_indices,
            )

        # Build dataset style 1 only
        dataset = PHC_style_subset(
            trafo,
            apply_symmetric_transforms=False,
            data_path=data_path,
            style=1,
            return_all_annotations=True,
        )
        train_split = int(np.floor(splitratio[0] * len(dataset)))
        val_split = int(np.floor(splitratio[1] * len(dataset)))
        test_split = int(np.floor(splitratio[2] * len(dataset)))
        indices = list(range(len(dataset)))
        if randomsplit == True:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices, val_indices, test_indices = (
            indices[:train_split],
            indices[train_split : val_split + train_split],
            indices[val_split + train_split :],
        )

        if split == "train":
            dataset_style1_train = Subset(
                PHC_style_subset(
                    trafo,
                    apply_symmetric_transforms=True,
                    data_path=data_path,
                    style=1,
                    return_all_annotations=True,
                ),
                train_indices,
            )

        if split == "val":
            dataset_style1_val = Subset(
                PHC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                    return_all_annotations=True,
                ),
                val_indices,
            )

        if split == "test":
            dataset_style1_test = Subset(
                PHC_style_subset(
                    trafo,
                    apply_symmetric_transforms=False,
                    data_path=data_path,
                    style=1,
                    return_all_annotations=True,
                ),
                test_indices,
            )

        # Concatenate the datasets
        if split == "train":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_train, dataset_style1_train]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "val":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_val, dataset_style1_val]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

        if split == "test":
            dataset = torch.utils.data.ConcatDataset(
                [dataset_style0_test, dataset_style1_test]
            )
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
            )

    return dataloader, dataset


if __name__ == "__main__":
    # dataloader1, _ = get_dataloader(task='isic3', split='train', batch_size=4)
    # dataloader2, _ = get_dataloader(task='isic3', split='val', batch_size=4)
    # dataloader3, _ = get_dataloader(task='isic3', split='test', batch_size=4)
    # print(f'Dataloader for Train Set ISIC has {len(dataloader1)} samples!')
    # print(f'Dataloader for Val Set ISIC has {len(dataloader2)} samples!')
    # print(f'Dataloader for Test Set ISIC has {len(dataloader3)} samples!')

    testloader, test = get_dataloader("phc_style_concat", "train", 5)
    print(len(test))

    sample = next(iter(testloader))
    image, mask, a_list, style = sample
    print(type(image))
    print(type(mask))
    print(type(a_list[1]))
    print(f"image has shape {image.size()}")
    print(f"mask has shape {mask.size()}")
    print(f"style has shape {style.size()}")
