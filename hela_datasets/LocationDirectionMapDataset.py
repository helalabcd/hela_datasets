import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torch.utils.data import Dataset
import os
from PIL import Image
from .helpers import gaussian_on_canvas, get_cell_centroids, get_centroid_map

import os.path
import pickle

import hashlib
import json
from .vector_helpers import process_burst

import warnings
warnings.filterwarnings("ignore")

class LocationDirectionMapDataset(Dataset):

    def __init__(self, base_path=None, centroid_size_sigma=1):

        # Create a hash of the config parameters above to make each class uniquely cache-able
        params_dict = {"base_path": base_path, "centroid_size_sigma": centroid_size_sigma}
        params_json = json.dumps(params_dict, sort_keys=True)
        self.config_hash = hashlib.sha256(params_json.encode()).hexdigest()

        assert base_path is not None, "Please provide base_path, it should usually be HeLa_dataset/train or HeLa_dataset/test"
        if base_path is None:
            print("ERROR")
            return
        
        self.centroid_size_sigma = centroid_size_sigma
        self.bursts = os.listdir(base_path)[:2]

        self.cached_bursts = []
        self.burst_start_index = []
        current_img_index_counter = 0
        
        os.makedirs("cache", exist_ok=True)
        for burst in tqdm(self.bursts):
            
            cache_base = "cache/aCACHE_LocationDirectionMapDataset" + str(self.config_hash)
            if not os.path.exists(cache_base):
                os.makedirs(cache_base)

            cache_path = f"{cache_base}/{burst}.cache.p"
            if not os.path.isfile(cache_path):
                print("Processing uncached burst", burst)
                
                burst_frame_infos = process_burst(base_path + "/" + burst, self.centroid_size_sigma)
                self.cached_bursts.append(burst_frame_infos)

                pickle.dump( burst_frame_infos, open( cache_path, "wb" ) )

            else:
                print("Loading cached burst", burst)
                burst_frame_infos = pickle.load( open( cache_path, "rb" ) )
                self.cached_bursts.append(burst_frame_infos)


    def __getitem__(self, idx):
        return self.cached_bursts[idx]

    def __len__(self):
        return len(self.self.cached_bursts)
        
