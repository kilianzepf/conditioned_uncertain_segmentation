from matplotlib.colors import ListedColormap
import numpy as np
import glob2 as glob
import PIL.Image as Image
import random


import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

"""
PhC Dataset
"""


class PHC(torch.utils.data.Dataset):
    def __init__(self, transform, apply_symmetric_transforms, data_path):
        # Initialization
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms

        # Image List
        # all image paths stored in a list
        image_paths = sorted(glob.glob(data_path + "/images/*"))
        annotator1_paths = sorted(glob.glob(data_path + "/fine1/*"))
        annotator2_paths = sorted(glob.glob(data_path + "/fine2/*"))
        annotator3_paths = sorted(glob.glob(data_path + "/fine3/*"))
        annotator4_paths = sorted(glob.glob(data_path + "/coarse1/*"))
        annotator5_paths = sorted(glob.glob(data_path + "/coarse2/*"))
        annotator6_paths = sorted(glob.glob(data_path + "/coarse3/*"))

        self.image_paths = 6 * image_paths
        self.all_annotator_paths = (
            annotator1_paths
            + annotator2_paths
            + annotator3_paths
            + annotator4_paths
            + annotator5_paths
            + annotator6_paths
        )

        self.annotator1_paths = 6 * annotator1_paths
        self.annotator2_paths = 6 * annotator2_paths
        self.annotator3_paths = 6 * annotator3_paths
        self.annotator4_paths = 6 * annotator4_paths
        self.annotator5_paths = 6 * annotator5_paths
        self.annotator6_paths = 6 * annotator6_paths

    def symmetric_augmentation(self, images_and_masks=[]):
        """
        Applies affine transformations for image and masks.

        Args:
            images_and_masks: (list of torch.tensors) images and masks to transform
        Returns:
            List of torch.tensors that are transformed in the same way (symmetric) and therefore suitable for semantic segmentation

        """

        # Random Horizontal Flip
        if np.random.random() > 0.5:
            images_and_masks = [TF.hflip(x) for x in images_and_masks]

        # Random Vertical Flip
        if np.random.random() > 0.5:
            images_and_masks = [TF.vflip(x) for x in images_and_masks]

        # Shift/Scale/Rotate Randomly
        angle = random.randint(-15, 15)
        translation = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        scale = random.uniform(0.9, 1.1)  # prev 0.9 1.1
        shear = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))

        images_and_masks = [
            TF.affine(
                x, angle=angle, translate=translation, scale=scale, shear=shear, fill=0
            )
            for x in images_and_masks
        ]

        return images_and_masks

    def get_pathnahme(self, idx):
        image_path = self.image_paths[idx]
        all_path = self.all_annotator_paths[idx]
        target1_path = self.annotator1_paths[idx]
        target2_path = self.annotator2_paths[idx]
        target3_path = self.annotator3_paths[idx]
        target4_path = self.annotator4_paths[idx]
        target5_path = self.annotator5_paths[idx]
        target6_path = self.annotator6_paths[idx]
        return (
            image_path,
            all_path,
            target1_path,
            target2_path,
            target3_path,
            target4_path,
            target5_path,
            target6_path,
        )

    def __len__(self):
        # Returns the total number of samples in the DataSet
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Generates one sample of data

        image_path = self.image_paths[idx]
        target_path = self.all_annotator_paths[idx]
        target1_path = self.annotator1_paths[idx]
        target2_path = self.annotator2_paths[idx]
        target3_path = self.annotator3_paths[idx]
        target4_path = self.annotator4_paths[idx]
        target5_path = self.annotator5_paths[idx]
        target6_path = self.annotator6_paths[idx]

        path_list = [
            image_path,
            target_path,
            target1_path,
            target2_path,
            target3_path,
            target4_path,
            target5_path,
            target6_path,
        ]

        def path_to_image(path):
            img = Image.open(path)  # Read in with Pillow
            return img

        # Prepares images
        def pil_image(x):
            x = path_to_image(x)
            # x = x.astype('int16')
            return x

        def to_tensor_trafo(x):
            x = self.transform(x)
            x = x.float()
            return x

        pil_image_list = [pil_image(x) for x in path_list]
        (
            image,
            target,
            target1,
            target2,
            target3,
            target4,
            target5,
            target6,
        ) = pil_image_list
        # Activate symmetric transformations for Data Augmentation, see method description of symmetric_augmentation
        if self.symmetric_transforms:
            pil_image_list = self.symmetric_augmentation(pil_image_list)

        final_list = [to_tensor_trafo(x) for x in pil_image_list]
        image, target, target1, target2, target3, target4, target5, target6 = final_list
        # Normalize each patch by its mean and variance and set intensities to 0 that are more then 3 stds away
        # mean = torch.mean(image)
        # std = torch.std(image)
        # image = (image-mean)/(3*std)
        # image = TF.normalize(image, torch.mean(image), torch.std(image))
        return image, target, [target1, target2, target3, target4, target5, target6]


if __name__ == "__main__":
    trafo = transforms.Compose([transforms.ToTensor()])
    data_path = "/home/kmze/conditioned_uncertain_segmentation/data/phc_data/phc_data"
    dataset = PHC(trafo, apply_symmetric_transforms=False, data_path=data_path)
    print(len(dataset))
