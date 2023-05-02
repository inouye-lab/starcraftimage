import pandas as pd
import numpy as np

def _postprocess_train_test_split(metadata, train, perc_train=0.9, random_state=1):
    # First filter to only matches that are in the labels
    metadata = pd.concat([filt_md for target_id, filt_md in _stratify_by_label(metadata)])
    match_metadata = metadata.drop_duplicates(subset=['replay_name']).reset_index(drop=True)

    # Split into train and test along unique matches
    n_match_train = int(np.round(perc_train * len(match_metadata)))
    perm = np.random.RandomState(random_state).permutation(len(match_metadata))
    if train == 'all':
        matches = match_metadata
    elif train == True:
        matches = match_metadata.iloc[perm[:n_match_train], :]
    elif train == False:
        matches = match_metadata.iloc[perm[n_match_train:], :]
    else:
        raise ValueError('`train` must be True, False or \'all\'')
    # Filter based on matches
    return _filter_by_matches(metadata, matches).sample(frac=1, random_state=0).reset_index(drop=True)  # Shuffle


def _filter_by_matches(metadata, match_metadata):
    metadata = metadata[metadata['replay_name'].isin(match_metadata['replay_name'])]
    return metadata.reset_index(drop=True)


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


def _stratify_by_label(md):
    # Get unique target ids
    unique_ids = md['target_id'].unique()
    # Get metadata filtered by target id
    for target_id in unique_ids:
        # Filter by target_id
        filt_md = md[md['target_id'] == target_id].reset_index(drop=True)
        # Yield this group
        yield target_id, filt_md


def _train_test_split_and_sample(md, train, n_train, n_test, random_state=0):
    '''Split into train and test based on match data. Then sample to exact n_train or n_test.'''
    # Split roughly into train and test by matches
    perc_train = n_train / (n_train + n_test)
    # NOTE: This returns train or test metadata already
    md = _postprocess_train_test_split(md, train, perc_train=perc_train)

    # Randomly sample exact number of windows
    perm = np.random.RandomState(random_state).permutation(len(md))

    if train == 'all':
        raise ValueError('For StarCraftMNIST and StarCraftCIFAR10 train=\'all\' is not an option')
    elif train == True:
        md = md.iloc[perm[:n_train], :]
    elif train == False:
        md = md.iloc[perm[:n_test], :]
    else:
        raise ValueError('`train` must be True, False or \'all\'')
    return md.reset_index(drop=True)
