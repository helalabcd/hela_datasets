from .LocationDirectionMapDataset import LocationDirectionMapDataset
import numpy as np
from torch.utils.data import Dataset
from .augmentation import FixedTransform
import torch
import math


class ForwardGridDataset(Dataset):

    def __init__(self, base_path=None, centroid_size_sigma=1, sequence_length=1, crop_size=128, device="cpu"):
        
        self.start_at = []
        self.total_length = 0
        self.sequence_length = sequence_length
        self.crop_size = crop_size
        self.device = device

        self.ds = LocationDirectionMapDataset(base_path, centroid_size_sigma=centroid_size_sigma)

        for burst in self.ds:
            X, (c,d1,d2) = burst

            self.start_at.append(self.total_length)
            self.total_length += len(X) - self.sequence_length
            

    def __getitem__(self, idx):
        
        start_idx = max([x for x in self.start_at if x <= idx])
        burst_index = self.start_at.index(start_idx)

        inburst_index = idx - start_idx

        X = self.ds[burst_index][0][inburst_index:inburst_index+self.sequence_length]
        centroids = self.ds[burst_index][1][0][inburst_index:inburst_index+self.sequence_length]
        y1 = self.ds[burst_index][1][1][inburst_index:inburst_index+self.sequence_length]
        y2 = self.ds[burst_index][1][2][inburst_index:inburst_index+self.sequence_length]
        
        fixed_transformation = FixedTransform(min_angle=0, max_angle=359, crop_height=self.crop_size, crop_width=self.crop_size)

        """ Apply Fixed Transform (needs a channel dimension, so we unsqueeze and squeeze again) """
        X = torch.Tensor(X)[:, None, :, :].to(self.device)
        centroids = torch.Tensor(centroids)[:, None, :, :].to(self.device)
        y1 = torch.Tensor(y1)[:, None, :, :].to(self.device)
        y2 = torch.Tensor(y2)[:, None, :, :].to(self.device)

        X = fixed_transformation(X)
        centroids = fixed_transformation(centroids)
        y1 = fixed_transformation(y1)
        y2 = fixed_transformation(y2)

        X = X.squeeze()
        centroids = centroids.squeeze()
        y1 = y1.squeeze()
        y2 = y2.squeeze()

        """ Arrange the frames and labels into a supergrid each (works best when sequence_lenght is a perfect square """

        def make_supergrid(a):
            num, x, y = a.shape
            l = math.ceil(math.sqrt(num))
            canvas = torch.zeros((l*x, l*y))

            pointer = 0
            for ix in range(l):
                for iy in range(l):
                    if pointer < len(a):
                        canvas[ix*x:(ix*x+x), iy*y:(iy*y+y)] = a[pointer]
                    else:
                        print("WARN: Your sequence length is not a perfect square!")
                    pointer += 1
            return canvas

        # Create supergrid, also add singular color channel
        X_supgrid = make_supergrid(X)[None]
        centroids_supgrid = make_supergrid(centroids)[None]
        y1_supgrid = make_supergrid(y1)[None]
        y2_supgrid = make_supergrid(y2)[None]

        return X_supgrid, (centroids_supgrid, y1_supgrid, y2_supgrid)

    def __len__(self):
        return self.total_length

