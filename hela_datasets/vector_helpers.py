import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import torch
import os

from collections import defaultdict

from .helpers import gaussian_on_canvas
from .helpers import get_cell_centroids, get_centroid_map

head = ["frame", "cellid", "a", "b", "c", "d", "xy","xz","xu","x"]

def get_inverse_mappings(burst="train/Burst1_A4_1_VesselID-29_2-0/"):

    #burst = "train/Burst1_A4_1_VesselID-29_2-0/"
    path = burst + "img1/"
    
    df = pd.read_csv(f"{burst}/gt/gt.txt", sep=",", names=head)
    path = burst + "img1/"
    images = [Image.open(path + x) for x in sorted(os.listdir(path))]

    all_frames = []
    for frame_index in range(len(images)):
        frame_index += 1
        this_frame_dict = defaultdict(dict)
        for cell_id in df[df["frame"] == frame_index]["cellid"].tolist():
    
            #row = df[df["frame"] == frame_index][df["cellid"] == cell_id]
            row = df[(df["frame"] == frame_index) & (df["cellid"] == cell_id)]

            a = row.a.values[0]
            b = row.b.values[0]
            c = row.c.values[0]
            d = row.d.values[0]
            a = int(a)
            b = int(b)
            c = int(c)
            d = int(d)

            this_frame_dict[cell_id] = (b+(d//2), a+(c//2))
        
        all_frames.append(this_frame_dict)
    return all_frames

def get_vector_mappings(burst="train/Burst1_A4_1_VesselID-29_2-0/"):

    inverse_mappings = get_inverse_mappings(burst)

    movements = {}
    last_mapping = None
    for frame_index, mapping in enumerate(inverse_mappings):
        movements[frame_index] = {}

        if last_mapping is None:
            last_mapping = mapping
            continue
        
        movement_x_component = {}
        movement_y_component = {}

        #print("mapping", mapping.keys())
        for cell_id in mapping.keys():
            print("cell_id", cell_id)
            #print("xd", cell_id, mapping[cell_id], last_mapping[cell_id])
            try:
                difference = mapping[cell_id][0]-last_mapping[cell_id][0], mapping[cell_id][1]-last_mapping[cell_id][1]
            except:
                print("error in difference calculation, setting difference to 0")
                difference = 0,0
            #print(difference)
            #print(frame_index)

    
            movements[frame_index-1][cell_id] = difference
        last_mapping = mapping

    return movements
        
#mm = get_vector_mappings()



def get_scaled_centroid_map(mask_img, scalars, centroid_size_sigma):
    gaussiansx = []
    gaussiansy = []
    
    for cell_id in np.unique(np.array(mask_img)):
        
        if cell_id == 0:
            # Background
            continue
        
        match_mask = np.array(mask_img) == cell_id
    
        # Find the indices where the array is True
        y_indices, x_indices = np.where(match_mask)
        
        # Calculate the average of the indices
        centroid_y = np.mean(y_indices)
        centroid_x = np.mean(x_indices)
    
        #print(cell_id, centroid_y, centroid_x)
        #print(scalars, scalars[cell_id][0])
        if int(cell_id) in scalars.keys():
            scalex = scalars[int(cell_id)][0]
            scaley = scalars[int(cell_id)][1]
            #print(scalex, scaley)
        else:
            scalex = 0
            scaley = 0
            #print("scaler for x not in keys", cell_id)

        goc_x = gaussian_on_canvas(mask_img.shape, mue=(centroid_y, centroid_x), sigma=(centroid_size_sigma,centroid_size_sigma))
        goc_y = gaussian_on_canvas(mask_img.shape, mue=(centroid_y, centroid_x), sigma=(centroid_size_sigma,centroid_size_sigma))

        #goc_x *= 1/goc_x.max()
        #goc_y *= 1/goc_y.max()
        
        gaussiansx.append(goc_x * scalex)
        gaussiansy.append(goc_y * scaley)

    a = np.mean(np.array(gaussiansx), axis=0).T
    a_index_of_most_extreme = np.argmax(np.abs(a))
    
    b = np.mean(np.array(gaussiansy), axis=0).T
    b_index_of_most_extreme = np.argmax(np.abs(b))
    
    return a, b




def process_burst(burst, centroid_size_sigma):

    if not burst.endswith("/"):
        burst = burst + "/"

    df = pd.read_csv(f"{burst}/gt/gt.txt", sep=",", names=head)
    path = burst + "img1/"
    vector_mappings = get_vector_mappings(burst)
    images = [Image.open(path + x).convert('L') for x in sorted(os.listdir(path))]

    frame_infos = []
    for frame_index, img in enumerate(images):
        frame_index += 1
        centroid_map = get_cell_centroids(img, df, frame_index)
        result = get_centroid_map(centroid_map, centroid_size_sigma)

        vectorx, vectory = get_scaled_centroid_map(centroid_map, vector_mappings[frame_index-1], centroid_size_sigma)

        frame_infos.append((result, vectorx, vectory))

    return frame_infos
