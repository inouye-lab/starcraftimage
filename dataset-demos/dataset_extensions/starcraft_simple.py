import PIL
import torch
import numpy as np
import pandas as pd

from sc2image.dataset import StarCraftImage
from sc2image.utils.metadata_processing import _stratify_by_label, _train_test_split_and_sample
from sc2image.dataset_extensions.modules import StarCraftToImageReducer

class StarCraftSimple(StarCraftImage):
    def __init__(self, 
                    root_dir, 
                    train=True,
                    image_format=None,  # Must be 'mnist' or 'cifar10'
                    transform=None,  # Transform applied to image
                    target_transform=None,  # Transform applied to label
                    use_metadata_cache=False,
                    download=False
                ):
        """
        A wrapper on StarCraftImage which produces simpler data representations such as the StarCraftMNIST
        and StarCraftCIFAR10 dataset. 
        Note: this is much slower than using the StarCraftMNIST and StarCraftCIFAR10 datasets directly as this 
        transforms each sample the its hyperspectral format to the desired format on the fly.
        Args:
            root_dir (str): Path to the root directory where the `starcraft-image-dataset` directory exists or will be
                saved to if download is set to True.
            train (bool): Whether to return the training or test set. If True, returns the training set. If False, 
                returns the test set. If 'all', returns the entire dataset.
            image_format (str): The desired image format. Must be 'mnist' or 'cifar10'.
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            use_metadata_cache: Loads the metadata from a cached file if it exists. If False, the metadata is
                loaded using the metadata.csv file (which is slower)
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
        """
        if image_format.lower() == 'mnist':
            self.is_mnist = True
            self.postprocess_metadata_fn = _postprocess_mnist
            image_size = 28
        elif image_format.lower() == 'cifar10':
            self.is_mnist = False
            self.postprocess_metadata_fn = _postprocess_cifar10
            image_size = 32
        else:
            raise ValueError(f'Unrecognized image_format: {image_format}. Must be "mnist" or "cifar10"')
        self._reduce_to_image = StarCraftToImageReducer()
        self.transform = transform
        self.target_transform = target_transform
        super().__init__(root_dir, train=train, image_size=image_size, image_format='bag-of-units',
                         use_metadata_cache=use_metadata_cache, return_label=True, label_kind='10-class',
                         return_dict=False, download=download)
        
    def _filter_metadata(self, md):            
        print('Post-processing metadata')
        assert self.postprocess_metadata_fn in [_postprocess_cifar10, _postprocess_mnist]
        md = self.postprocess_metadata_fn(md, train=self.train)
        md = md.reset_index(drop=True)  # Renumber rows
        md['10_class_data_split'] = self.data_split
        return md
        

    def __getitem__(self, idx):
        x, target = self._get_x_and_target(idx)
        if self.is_mnist:
            x = self._flatten_to_mnist(x)

        # Return as PIL image
        if x.shape[0] == 1:  # Grayscale
            img = PIL.Image.fromarray(x.squeeze(0).type(torch.uint8).numpy(), mode='L')
        elif x.shape[0] == 3:  # RGB
            img = PIL.Image.fromarray(x.permute(1,2,0).type(torch.uint8).numpy(), mode='RGB')
        else:
            raise RuntimeError('x should have 1 or 3 channels')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _get_x_and_target(self, idx):
        # Get bag-of-units representation
        (bag_of_units_ids, bag_of_units_values), target = super().__getitem__(idx)
        # Flatten to image
        x = self._reduce_to_image(bag_of_units_ids.unsqueeze(0), bag_of_units_values.unsqueeze(0))[0]
        return x, target


    def _flatten_to_mnist(self, x):
        # Reorder RGB to get in terms of player 1, player 2 and neutral
        #  and get corresponding masks
        x_ordered = x[[2,0,1], :, :] / 255.0
        m1, m2, mn = (x_ordered > 0)

        # Normalize neutral
        n = x_ordered[2, :, :]
        x_ordered[2, :, :] = (n - n.min()) / (n.max() - n.min())

        # Rescale values
        # NOTE: for player 2 it is 0.4 to 0 (so it becomes a negative multiplier)
        scales = np.array([[0.55, 1], [0.45, 0], [0.48, 0.52]])
        v1, v2, vn = [(sc[0] + (sc[1] - sc[0]) * v) for v, sc in zip(x_ordered, scales)]

        new_x = (vn * (~m2) + v2 * m2) * (~m1) + v1 * m1
        return 255.0 * new_x.unsqueeze(0)


def _postprocess_cifar10(*args, **kwargs):
    n_train = kwargs.get('n_train', 5000)  # number of training samples **per class**
    n_test = kwargs.get('n_test', 1000)  # number of test samples **per class**
    return _postprocess_simplified(
        *args, **kwargs,
        n_train=n_train, n_test=n_test)


def _postprocess_mnist(*args, **kwargs):
    n_train = kwargs.get('n_train', 6000)  # number of training samples **per class**
    n_test = kwargs.get('n_test', 1000)  # number of test samples **per class**
    return _postprocess_simplified(
        *args, **kwargs,
        n_train=n_train, n_test=n_test)


def _postprocess_simplified(metadata, train, n_train, n_test):
    '''Filter metadata via stratified sampling. 
    First stratify based on class. 
    Then split based on matches. 
    Finally, sample without replacement to get exact numbers.'''
    # drop all rows which are not in the 10 class data split
    metadata.dropna(subset=['10_class_data_split'], inplace=True)
    metadata.reset_index(drop=True, inplace=True)
    metadata.drop(columns=['14_class_data_split'], inplace=True)
    return pd.concat([
        _train_test_split_and_sample(
            filt_md, train, n_train=n_train, n_test=n_test, random_state=np.abs(int(target_id)))
                for target_id, filt_md, in _stratify_by_label(metadata)
    ]).sample(frac=1, random_state=0).reset_index(drop=True)  # Shuffle rows