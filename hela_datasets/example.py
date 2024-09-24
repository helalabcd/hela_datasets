from __init__ import LocationMapDataset
from __init__ import LocationDirectionMapDataset

import matplotlib.pyplot as plt

ds = LocationDirectionMapDataset("HeLa_dataset/train", centroid_size_sigma=2)

assert len(ds[0][0]) == 3

fig, ax = plt.subplots(1,3)

ax.flat[0].imshow(ds[1][0][0])
ax.flat[1].imshow(ds[1][0][1])
ax.flat[2].imshow(ds[1][0][2])
plt.show()




ds = LocationDirectionMapDataset("HeLa_dataset/train", centroid_size_sigma=6)
assert len(ds[0][0]) == 3

fig, ax = plt.subplots(1,3)

ax.flat[0].imshow(ds[1][0][0])
ax.flat[1].imshow(ds[1][0][1])
ax.flat[2].imshow(ds[1][0][2])
plt.show()