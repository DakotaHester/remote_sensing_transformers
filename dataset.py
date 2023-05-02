# using torchgeo for chesapeake bay 13 class
import os
import matplotlib.pyplot as plt
from torchgeo.datasets import ChesapeakeCVPR, RasterDataset
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler
from torch.utils.data import DataLoader


def getChesapeakeDataset():
    cblc = ChesapeakeCVPR(
        root='data',
        splits=['ny-train', 'ny-val', 'ny-test'],
        layers=['naip-new', 'lc'],
        transforms=None,
        cache=True,
        download=True,
        checksum=False,
    )

def getNycData(dir='data/source'):
    class NYC_Imagery(RasterDataset):
        filename_glob = 'ortho_*.tif'
        # filename_regex = r''
        # date_format = "%Y%m%dT%H%M%S"
        is_image = True
        separate_files = False
        all_bands = ["NIR", "BLUE", "GREEN"]
        
    class NYC_Mask(RasterDataset):
        filename_glob = 'landcover_5.tif'
        # filename_regex = r''
        # date_format = "%Y%m%dT%H%M%S"
        is_image = False
        # separate_files = False
        # all_bands = ["NIR", "BLUE", "GREEN"]

    imagery = NYC_Imagery(
        root=dir,
        cache=True
    )

    masks = NYC_Mask(
        root=dir,
        cache=True
    )
    
    return imagery & masks # intersection dataset

getNycData()