import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import resize
import numpy as np
import sparse as np_sparse
from PIL import Image


def project_for_plotting(sparse_mat, reshape=None, reduction='max'):
    "Sums a 3d sparse matrix along the 0th dim to project it down to a 2d sparse tensor (and clips for plotting)"
    assert sparse_mat.ndim == 3
    if isinstance(sparse_mat, (torch.Tensor, )):
        if reduction == 'sum':
            if torch.is_floating_point(sparse_mat):
                out = torch.clip(torch.sparse.sum(sparse_mat, dim=0), 0.0, 1.0)
            else:
                out = torch.clip(torch.sparse.sum(sparse_mat, dim=0), 0, 255)
        else:
            raise NotImplementedError('torch has no max')
        out = out.to_dense()
    else:
        # assume it's a np_sparse.COO tensor
        if reduction == 'sum':
            if sparse_mat.dtype == int:
                out = sparse_mat.sum(axis=0).clip(0, 255)
            else:
                out = sparse_mat.sum(axis=0).clip(0, 1.0).astype(float)
        else:
            out = sparse_mat.max(axis=0)
        out = out.todense()

    if reshape:
        out = resize(out[None, None, :, :], reshape).squeeze()
    return out

def make_plots(images, titles=None, show=True, axes=None, axis='off'):
    if axes is None:
        fig, axes = plt.subplots(1, len(images), figsize=(3*len(images), 5))
    else:
        assert axes.ndim == 1, 'Axes must be 1d'
    for image_idx, image in enumerate(images):
        if titles is not None:
            if isinstance(image, (torch.Tensor, )):
                if image.is_sparse:
                    if image.shape[0] > 3:
                        image = project_for_plotting(image)
                    else:
                        image = image.to_dense()
            else:
                if isinstance(image, (np_sparse.COO)):
                    if image.shape[0] > 3:
                        image = project_for_plotting(image)
                    else:
                        image = image.to_dense()
                    
            imshow(image, ax=axes[image_idx], title=titles[image_idx])
        else:
            imshow(image, ax=axes[image_idx])

    for ax in axes:
        ax.axis(axis)

    if show and axes is None:
        plt.tight_layout()
        plt.show()


def imshow(image, ax=None, title=None, **kwargs):
    # A simple helper function for plotting images with (C, H, W) format instead of (H, W, C)
    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    if isinstance(image, Image.Image):  # checking if PIL image
        ax.imshow(image)
            
    elif isinstance(image, torch.Tensor):
        if image.is_sparse:
            image = image.to_dense()
    
    elif isinstance(image, (np_sparse.COO, )):
        image = image.todense()
        
    if len(image.shape) == 2 or image.shape[0] == 1:
        ax.imshow(image.squeeze(), cmap='gray')

    elif image.shape[0] == 3:  # if image.shape = (C, H, W)
        if isinstance(image, torch.Tensor):
            ax.imshow(torch.moveaxis(image, 0, -1))
        else:
            ax.imshow(np.moveaxis(image, 0, -1))
    else:  # if image.shape = (H, W, C), i.e. no channel moving needed
        ax.imshow(image)

    ax.set_title(title)
    return ax