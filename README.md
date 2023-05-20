# StarCraftImage Dataset

Welcome! This is the repository for the StarCraftImage dataset from the paper: `StarCraftImage: A Dataset For Prototyping Spatial Reasoning
Methods For Multi-Agent Environments`

## Quickstart
To use the dataset, you can import it and instantiate the dataset using the following:

```
from sc2image.dataset import StarCraftImage
scimage = StarCraftImage(root_dir=<your_download_path>, download=True)
```
This will download the StarCraftImage dataset to the `<your_download_path>` directory (if it does not already exist there).
As this dataset has over 3.6 million samples, this might take a while (see CIFAR10 and MNIST sections below for a quicker download).

## Example uses
Please see the demo notebooks in the `dataset-demos` folder to see different formats which you can use this dataset!

Cheers!