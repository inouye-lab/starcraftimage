import os
import collections
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torchvision.transforms.functional
from torchvision import io
from torchvision.datasets.utils import download_and_extract_archive
import PIL

from .modules import StarCraftToImageReducer
from .utils.unit_type_data import NEUTRAL_IDS as neutral_ids, NONNEUTRAL_IDS as nonneutral_ids, NO_UNIT_CHANNEL
from .utils.metadata_processing import _postprocess_cifar10, _postprocess_mnist

EXTRACTED_IMAGE_SIZE = 64
TABULAR_KEYS = ['minerals', 'vespene', 
                'food_used', 'food_cap', 'food_army', 'food_workers', 
                'idle_worker_count', 'army_count', 'warp_gate_count', 
                'larva_count']
TERRAIN_KEYS = ['pathing_grid', 'placement_grid', 'terrain_height']

RACE_TO_ID = {'Terran':1, 'Zerg': 2, 'Protoss': 3}
ID_TO_RACE = {v:k for k, v in RACE_TO_ID.items()}

_DEFAULT_10_LABELS_DICT = {
    0: ('Acolyte LE', 'Beginning'),
    1: ('Acolyte LE', 'End'),
    2: ('Abyssal Reef LE', 'Beginning'),
    3: '(Abyssal Reef LE, End)',
    4: ('Ascension to Aiur LE', 'Beginning'),
    5: ('Ascension to Aiur LE', 'End'),
    6: ('Mech Depot LE', 'Beginning'),
    7:('Mech Depot LE', 'End'),
    8: ('Odyssey LE', 'Beginning'),
    9: ('Odyssey LE', 'End')
}


################################ #TODO ################################
#######################################################################
"""
* How are we going to handle samples which don't have a y label (e.g., the two maps which are not included in the default 10 classes)? 
  If drop them by default, this will by default drop all samples from 2/7 maps

* Fill in docstrings for all functions



"""
#######################################################################
#######################################################################




class StarCraftImage(torch.utils.data.Dataset):
    '''
    Given a root data directory, create a StarCraft dataset (downloading if necessary)
    from metadata and replay png files in that directory.

    Params
    ------
    `use_sparse` : If False (default), then return dense tensors
                   for unit information.  One is '{prefix}_unit_ids'
                   and the other is '{prefix}_values'. Both have
                   shape (C', W, H) where C' is the number of
                   overlapped units for one xy coordinate.
                   Importantly, this is variable for each
                   instance and thus must be padded if batched
                   together with other samples (see included 
                   dataloader for this functionality).

                   If True, then return sparse PyTorch tensors
                   with shape (C, W, H) where C is the number of
                   unit types (different for players vs neutral).

    `to_float` :   Makes all return values except '{prefix}_unit_ids'
                   float type via `float()`. 'unit_ids' will
                   remain as LongTensor so they can be used with
                   embedding layers.
    '''

    def __init__(self,
                    root_dir='data',
                    train=True,  # can be True, False, or 'all', where 'all' yields both train and test
                    image_size=64,
                    image_format='dense-hyperspectral-image', # other formats are 'dense-bag-of-units' and 'sparse-hyperspectral-image'
                    to_float=True, 
                    drop_na=True,  # Drop samples which have no default y labels
                    return_y=False,
                    return_dict=False,
                    use_cache=False,
                    download=True,
                    ######### REMOVE BELOW #######
                    ):
        
        self._versions_dict = {  # Note, this only includes major and minor versions
            '1.0': {'download_url': 'xxx'}
        }
        self.dataset_name = 'starcraft-image-dataset'
        
        self.data_dir = self._initialize_data_dir(root_dir, download)
        assert train in [True, False, 'all'], f'train must be True, False, or "all" but got {train}'
        self.train = train
        self.data_split = 'train' if train==True else 'test' if train==False else 'all'
        self.to_float = to_float
        self.return_dict = return_dict
        self.return_y = return_y
        self.image_size = image_size

        # Load and process metadata (e.g., splitting to train or test)
        self.metadata = self._process_metadata(self._load_metadata(use_cache), drop_na)

    def _load_metadata(self, use_cache):
        """
        Load metadata from csv file, optionally using a cached version. 
        """
        if use_cache:
            md_cache_path = Path(self.data_dir) / 'cached-metadata.pkl'
            if md_cache_path.exists():
                print('Loading cached metadata found at ', str(md_cache_path))
                md = pd.read_pickle(md_cache_path)
            else:
                print('No cached metadata found at ', str(md_cache_path))
                print('Loading metadata from csv and saving to cache')
                md = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'), dtype={'target_id': 'Int64'})
                md.to_pickle(md_cache_path)
        else:
            print('Loading metadata from csv. Note: to speed this up in the future, set `use_cache=True`')
            md = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'), dtype={'target_id': 'Int64'})
        return md
    
    def _process_metadata(self, md, drop_na):
        """
        Filter metadata to train, test, or all (set by `train`), and drop samples with no y label (if `drop_na`)."""
        # Filter metadata to train, test, or all
        if self.data_split != 'all':
            md = md[md['data_split'] == self.data_split].reset_index(drop=True)  # split md to train/test/all
        if drop_na:
            md = md.dropna(subset=['target_id']).reset_index(drop=True)
        return md

    def _clear_cache(self):
        md_cache_path = Path(self.data_dir) / 'cached-metadata.pkl'
        if md_cache_path.exists():
            md_cache_path.unlink()
            print('Deleted cached metadata at ', str(md_cache_path))
        else:
            print('No cached metadata found at ', str(md_cache_path))

    def _initialize_data_dir(self, root_dir, download_flag):
        """
        Initialize the data directory, downloading the dataset if necessary
        """
        latest_version = sorted(list(self._versions_dict.keys()))[-1]  # getting the latest version

        root_dir = Path(root_dir)
        root_dir.mkdir(exist_ok=True)
        data_dir = root_dir / f'{self.dataset_name}_v{latest_version.replace(".", "_")}'
        # see if the dataset already exists
        if data_dir.exists() and len(os.listdir(data_dir)) > 0:
            print('Dataset found in ', str(data_dir))
        else:
            # the dataset does not exist, download it if download_flag is set to true
            if not download_flag:
                raise FileNotFoundError(
                    f'The {self.dataset_name} dataset could not be found in {data_dir}. Initialize the dataset with '
                    f'download=True to download the dataset.'
                )
            
            print(f'Dataset not found in {data_dir}, downloading...')
            data_dir.mkdir(exist_ok=True)
            try:
                download_and_extract_archive(
                    url=self._versions_dict[latest_version]['download_url'],
                    download_root=data_dir,
                    filename='starcraftimage.tar.gz',
                    remove_finished=True)
            except Exception as e:
                print(f'Download failure.\n{os.path.join(data_dir, "starcraftimage.tar.gz")} may be corrupted.',
                       'Please try deleting it and rerunning this command.\n')
                print(f'Exception: ', e)
        return data_dir

    def __str__(self):
        item = self[0]
        out = '-----------------\n'
        out += f'  {self.dataset_name}\n'
        out += f'  data_dir = {self.data_dir}\n'
        out += f'  data_split = {self.data_split}\n'
        out += f'  num_windows = {len(self)}\n'
        out += f'  num_matches = {self.num_matches()}\n'
        out += f'  image_size = ({self.image_size}, {self.image_size})\n'
        if type(item) is tuple:
            out += f'  getitem = {self[0]}\n'
        elif type(item) is dict:
            out += f'  getitem_keys = {self[0].keys()}\n'
        else:
            out += f'  getitem_type = {type(self[0])}\n'
        out += '-----------------'
        return out
        
    def __len__(self):
        return len(self.metadata)

    def num_matches(self):
        return len(self.md['replay_name'].unique())
    
    def __getitem__(self, idx):
        # Get necessary metadata
        window_png_filepath = self._get_window_png_path(idx)
        player_window_dict = self._convert_bag_png_to_player_dense_bag_window_dict(window_png_filepath)

        if self.use_sparse:
            # convert player_dense_bag_window_dict to sparse window dict
            player_window_dict = self._convert_player_dense_bag_window_dict_to_player_sparse_window_dict(player_window_dict)
        elif self.image_size != EXTRACTED_IMAGE_SIZE:
            # resize the images in the player_window_dict
            player_window_dict = self._resize_dense_player_window_dict(player_window_dict)
             

        data_dict = dict(
            **player_window_dict,
            **self._get_unit_tabular(idx),
            **self._get_non_unit_items(idx)
        ) 
        return self._check_float(data_dict)
    
    def _get_unit_tabular(self, idx):
        def _get_player_tabular(md_row, player_id):
            return [md_row[f'player_{player_id}_{key}'] for key in TABULAR_KEYS]
        
        md_row = self.metadata.iloc[idx]
        return dict(
            player_1_tabular = torch.tensor(_get_player_tabular(md_row, 1)),
            player_2_tabular = torch.tensor(_get_player_tabular(md_row, 2))
        )

    def _get_non_unit_items(self, idx):
        # Tuples of strings can be used in dictionaries
        md_row = self.metadata.iloc[idx]  # Single metadata entry
        return dict(
            # Extract terrain information   ('pathing_grid', 'placement_grid', 'terrain_height')
            **self._extract_terrain_info(idx, return_dict=True),
            # Extract target information
            is_player_1_winner = self._get_is_player_1_winner(idx),
            target_id = torch.tensor(md_row['target_id']),
        )

    def _check_float(self, data_dict):
        def _try_float(k, v):
            if k.endswith('unit_ids') or k in ['target_id', 'is_player_1_winner']:
                return v.to(torch.int64)
            try:
                v = v.float()
            except AttributeError:
                pass  # Ignore if not tensor
            return v
        if self.to_float:
            # Preserve LongTensor of unit_ids tensors otherwise convert
            return {k:_try_float(k, v) 
                    for k, v in data_dict.items()}
        else:
            return data_dict

    def _get_window_png_path(self, idx):
        md_row = self.metadata.iloc[idx]
        png_basename = f'idx_{md_row["global_idx"]}__replay_{md_row["replay_name"]}__window_{md_row["window_idx"]}.png'
        png_filepath = os.path.join(self.data_dir, 'png_files', png_basename)
        return png_filepath

    def _get_is_player_1_winner(self, idx):
        item = self.metadata.iloc[idx]
        return torch.tensor(item['winning_player_id'] == 1)
    
    def _convert_bag_png_to_player_dense_bag_window_dict(self, png_file_path):
        KEYS_OF_INTEREST = ['unit_values', 'unit_ids', 'map_state']
        player_prefix_to_channel_idx = {
            'player_2': 0,
            'neutral': 1,
            'player_1': 2}    

        dense_bag = io.read_image(str(png_file_path)).squeeze()
        # unstack the dense_bag back into channels
        assert dense_bag.shape[1] % EXTRACTED_IMAGE_SIZE == 0, f'Dense bag from png {png_file_path} is not evenly divided by image size: {self.image_size}'
        window_dict = {}
        for row_idx, key in enumerate(KEYS_OF_INTEREST):
            for player_prefix in ['player_1', 'player_2', 'neutral']:
                if key == 'map_state' and player_prefix == 'neutral':
                    continue
                channel_idx = player_prefix_to_channel_idx[player_prefix]
                stack = torch.stack(
                    [item for item in torch.split(
                        dense_bag[channel_idx,
                                  row_idx*EXTRACTED_IMAGE_SIZE:(row_idx+1)*EXTRACTED_IMAGE_SIZE], EXTRACTED_IMAGE_SIZE, dim=1)
                            if torch.any(item)], dim=0)
                window_dict[f'{player_prefix}_{key}'] = stack
        return window_dict
    
    def _convert_dense_player_bag_to_resized_player_hyperspectral(self, dense_bag_window_dict, player_prefix,
                                                                  return_sparse_tensor=True):
        non_empty_mask = dense_bag_window_dict[f'{player_prefix}_unit_ids'] != NO_UNIT_CHANNEL
        # getting indicies
        xy_idxs = non_empty_mask.nonzero()[:, 1:]  # getting the x,y spatial coordinates for the p1_uids
        c_idxs = dense_bag_window_dict[f'{player_prefix}_unit_ids'][non_empty_mask]
        idxs = torch.hstack((c_idxs.unsqueeze(1), xy_idxs))
        # getting values
        values = dense_bag_window_dict[f'{player_prefix}_unit_values'][non_empty_mask]

        n_channels = len(nonneutral_ids) if player_prefix != 'neutral' else len(neutral_ids)
        idxs, values, shape = self._resize_hyper(idxs, values,
                                                (n_channels, EXTRACTED_IMAGE_SIZE, EXTRACTED_IMAGE_SIZE))
        if return_sparse_tensor:
            return torch.sparse_coo_tensor(indices=idxs.T, values=values,
                                            size=shape).coalesce()
        else:
            return idxs, values, shape

    def _convert_player_dense_bag_window_dict_to_player_sparse_window_dict(self, dense_bag_window_dict):
        player_sparse_window_dict = {}
        # first calculate player hyperspectrals
        for player_prefix in ['player_1', 'player_2', 'neutral']:
            player_sparse_window_dict[f'{player_prefix}_hyperspectral'] = \
                self._convert_dense_player_bag_to_resized_player_hyperspectral(dense_bag_window_dict, player_prefix)
            if player_prefix != 'neutral':
                # the map state information is saved as a dense tensor, so just copy that over
                player_sparse_window_dict[f'{player_prefix}_map_state'] = dense_bag_window_dict[f'{player_prefix}_map_state']

        return player_sparse_window_dict

    def _resize_hyper(self, indices, values, shape):
        if self.image_size == EXTRACTED_IMAGE_SIZE:
            return indices, values, shape
        assert self.image_size <= EXTRACTED_IMAGE_SIZE, f'image_size must be less than or equal to {EXTRACTED_IMAGE_SIZE}'
        scale = float(self.image_size) / EXTRACTED_IMAGE_SIZE

        # Convert from index to location, then scale, then floor to get new index
        indices = np.array(indices)  
        values = np.array(values)
        indices[:, 1:] = (np.floor(scale * (indices[:, 1:] + 0.5))).astype(np.uint8)
    
        # How to handle overlaps (max coalesce - take max value for overlaps)
        # First sort (descending) by indices and then values
        #sort_idx = np.flip(np.lexsort((values, indices[:, 2], indices[:, 1], indices[:, 0]))) # Sort descending
        sort_idx = np.flip(np.argsort(values))  # Sort descending
        indices = indices[sort_idx, :]
        values = values[sort_idx]

        # Do unique over indices 
        unique_indices, unique_first_idx = np.unique(indices, axis=0, return_index=True)
        # Get corresponding values (which are the max because of sorting)
        unique_values = values[unique_first_idx]

        # Replace image_size in shape
        shape = (shape[0], self.image_size, self.image_size)

        # Convert to sparse tensor and coalesce, then extract new indices and values
        temp_sparse = torch.sparse_coo_tensor(unique_indices.T, unique_values, shape).coalesce()
        return temp_sparse.indices().T, temp_sparse.values(), shape
    
    def _convert_player_hyper_idxs_values_to_dense_bag_uids_and_uvalues(self, indices, values, shape):
            # Sort by xy coordinates so that np.split can be used later
            sort_idx = np.lexsort((indices[:, 2], indices[:, 1]))
            indices = indices[sort_idx, :]
            values = values[sort_idx]

            # Now can group by via splitting on first indices as returned by
            #  the "index" output of np.unique
            sorted_xy = indices[:, 1:]
            sorted_id = indices[:, 0]
            unique_xy, unique_first_idx = np.unique(sorted_xy, axis=0, return_index=True)
            ids_group_by_xy = np.split(sorted_id, unique_first_idx[1:])
            values_group_by_xy = np.split(values, unique_first_idx[1:])

            # Create new indices + values matrix with values of
            #  c', w, h, t, v  where c' are new channels (<= num_overlap), 
            #  w is x coord, h is y coord, t is unit type id, v is timestamp value
            def create_data(xy, ids, vals):
                channel_idx = np.arange(len(ids)).reshape((-1, 1))  # New channel dimension
                xy_idx = np.broadcast_to(xy, (len(ids), 2))  # xy dimensions replicated
                ids = ids.reshape((-1, 1)) # Types (which will serve as values in the dense type tensor)
                vals = vals.reshape((-1, 1))
                return np.hstack([channel_idx, xy_idx, ids, vals])
            
            new_data = np.vstack([
                create_data(xy, ids, vals)
                for xy, ids, vals in zip(unique_xy, ids_group_by_xy, values_group_by_xy)
            ])
            
            dense_shape = (new_data[:, 0].max() + 1, *shape[-2:])  # C' x W x H
            
            unit_ids = torch.sparse_coo_tensor(new_data[:,:3].T, new_data[:,3], size=dense_shape)
            unit_ids = unit_ids.to_dense()
            unit_values = torch.sparse_coo_tensor(new_data[:,:3].T, new_data[:,4], size=dense_shape)
            unit_values = unit_values.to_dense()
            return unit_ids, unit_values

    def _resize_dense_player_window_dict(self, dense_player_window_dict):
        if self.image_size == EXTRACTED_IMAGE_SIZE:
            return dense_player_window_dict
        assert self.image_size <= EXTRACTED_IMAGE_SIZE, f'image_size must be less than or equal to {EXTRACTED_IMAGE_SIZE}'
        # since the player values are essentially integer keys to a unit_id/unit_value lookup table, we need to resize carefully
        # to do this, we will use the _resize_hyper function to resize based on the unit_ids and unit_values
        for player_prefix in ['player_1', 'player_2', 'neutral']:
            idxs, values, shape = self._convert_dense_player_bag_to_resized_player_hyperspectral(dense_player_window_dict,
                                                                                                 player_prefix,
                                                                                                 return_sparse_tensor=False)
            # now we need to convert the idxs and values back into a dense tensor
            dense_player_window_dict[f'{player_prefix}_unit_ids'], dense_player_window_dict[f'{player_prefix}_unit_values'] = \
                self._convert_player_hyper_idxs_values_to_dense_bag_uids_and_uvalues(idxs.numpy(), values.numpy(), shape)
        return dense_player_window_dict

    def _extract_terrain_info(self, idx, return_dict=True):
        map_name = self.metadata.iloc[idx]['map_name']
        terrain_filepath = os.path.join(self.data_dir, 'map_png_files', map_name + '.png')
        terrain_images = io.read_image(terrain_filepath).squeeze()

        if self.image_size != terrain_images.shape[1]:
            terrain_images = torchvision.transforms.functional.resize(terrain_images,
                                                                (self.image_size, self.image_size)).type(torch.uint8)
        terrain_images = dict(
            pathing_grid=terrain_images[0] != 0,  # pathing_grid is binary, so convert
            placement_grid=terrain_images[1] != 0,  # placement_grid is binary, so convert
            terrain_height=terrain_images[2]  # terrain_height is continuous
        )
        if return_dict:
            return terrain_images
        else:
            return terrain_images['pathing_grid'], terrain_images['placement_grid'], terrain_images['terrain_height']
    

class StarCraftHyper(StarCraftImage):
    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, **kwargs)

class _StarCraftSimpleBase(StarCraftImage):
    def __init__(self, 
                    data_dir, 
                    train=True,
                    postprocess_metadata_fn=None,  # Function to apply to metadata after loading
                    transform=None,  # Transform applied to image
                    target_transform=None,  # Transform applied to label
                    image_size=64,
                    use_cache=False,
                ):
        self.postprocess_metadata_fn = postprocess_metadata_fn
        self._reduce_to_image = StarCraftToImageReducer()
        self.transform = transform
        self.target_transform = target_transform
        super().__init__(data_dir, train=train, image_size=image_size, use_cache=use_cache)
        
    def _process_metadata(self, md, drop_na):            
        if drop_na:
            md = md.dropna(subset=['target_id']).reset_index(drop=True)
        print('Post-processing metadata')
        assert self.postprocess_metadata_fn in [_postprocess_cifar10, _postprocess_mnist]
        md = self.postprocess_metadata_fn(md, train=self.train)
        md = md.reset_index(drop=True)  # Renumber rows
        md['data_split'] = self.data_split
        return md
        

    def __getitem__(self, idx):
        x, target = self._get_x_and_target(idx)

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
        # Get hyperspectral dictionary
        d = super().__getitem__(idx)
        # Create batch of size 1 and then extract output
        x = self._reduce_to_image(sc_collate([d]))[0]
        # Get target id directly from already extracted dictionary
        target = d['target_id']
        return x, target


class StarCraftCIFAR10(_StarCraftSimpleBase):
    def __init__(self, data_dir, train=True, transform=None, target_transform=None, use_cache=False):
        super().__init__(data_dir, train=True, image_size=32, postprocess_metadata_fn=_postprocess_cifar10,
                         transform=transform, target_transform=target_transform, use_cache=False)


class StarCraftMNIST(_StarCraftSimpleBase):
    def __init__(self, data_dir, train=True, transform=None, target_transform=None, use_cache=False):
        super().__init__(data_dir, train=True, image_size=28, postprocess_metadata_fn=_postprocess_mnist,
                         transform=transform, target_transform=target_transform, use_cache=False)

    def _get_x_and_target(self, idx):
        x, target = super()._get_x_and_target(idx)

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
        return 255.0 * new_x.unsqueeze(0), target

def starcraft_dense_ragged_collate(batch):
    '''
    Function to be passed as `collate_fn` to torch.utils.data.DataLoader
    when using use_sparse=False (default) for StarCraftImage.
    This handles padding the dense tensors so they have the same shape
    in each batch.

    `sc_collate` is an alias for this function as well.

    Example:
    >>> scdata = StarCraftImage(data_dir, use_sparse=False)
    >>> torch.utils.data.DataLoader(scdata, collate_fn=sc_collate, batch_size=32, shuffle=True)
    '''
    elem = batch[0]
    elem_type = type(elem)
    assert isinstance(elem, collections.abc.Mapping), 'Only works for dictionary-like objects'
    
    def pad_as_needed(A, n_target_channels):
        channel_pad = n_target_channels - A.shape[0]
        if channel_pad > 0:
            #A = np.pad(A, ((0, channel_pad), (0, 0), (0, 0)), mode='minimum')
            A = torch.nn.functional.pad(A, ((0, 0, 0, 0, 0, channel_pad)), value=A.min())
        return A
    
    def collate_pad(batch_list):
        # Pad each to have the same number of first dimension (i.e. channels for hyperspectral)
        try:
            ndim = batch_list[0].ndim
        except AttributeError:
            ndim = 0 # For non-tensors
        if ndim > 0:
            unique_channels = np.unique([d.shape[0] for d in batch_list], axis=0)
            if len(unique_channels) > 1:
                n_target_channels = unique_channels.max()
                batch_list = [pad_as_needed(d, n_target_channels) for d in batch_list]
        return torch.utils.data.dataloader.default_collate(batch_list)
    
    return elem_type({key: collate_pad([d[key] for d in batch]) for key in elem}) 


# Shorter alias for starcraft_dense_ragged_collate
sc_collate = starcraft_dense_ragged_collate
