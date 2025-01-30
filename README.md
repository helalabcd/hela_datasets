## Usage
```python
pip install git+https://github.com/helalabcd/hela_datasets
```

```python
from hela_datasets import LocationDirectionMapDataset

ds = LocationDirectionMapDataset("HeLa_dataset/train")

for burst in ds:
    x, y = burst
```


# Example: Interpretation of the per-burst dataset
![image](https://github.com/user-attachments/assets/15641613-8049-48aa-a616-5695d0f6b214)


fig, ax = plt.subplots(10, 4, figsize=(12, 24))

ds = LocationDirectionMapDataset("dummyhela/", centroid_size_sigma=8)

cmap_diverging = "seismic"

for burst in ds:
    x, (c, d1, d2) = burst

    for idx in range(10):
        # Normal image
        im1 = ax.flat[idx*4 + 0].imshow(x[idx])
        plt.colorbar(im1, ax=ax.flat[idx*4 + 0])

        # Normal image
        im2 = ax.flat[idx*4 + 1].imshow(c[idx])
        plt.colorbar(im2, ax=ax.flat[idx*4 + 1])

        # Use diverging colormap with a darker zero-point
        norm1 = mcolors.TwoSlopeNorm(vmin=-abs(d1).max(), vcenter=0, vmax=abs(d1).max())
        im3 = ax.flat[idx*4 + 2].imshow(d1[idx], cmap=cmap_diverging, norm=norm1)
        plt.colorbar(im3, ax=ax.flat[idx*4 + 2])

        norm2 = mcolors.TwoSlopeNorm(vmin=-abs(d2).max(), vcenter=0, vmax=abs(d2).max())
        im4 = ax.flat[idx*4 + 3].imshow(d2[idx], cmap=cmap_diverging, norm=norm2)
        plt.colorbar(im4, ax=ax.flat[idx*4 + 3])

    plt.savefig("viz.png")
```
![image](https://github.com/user-attachments/assets/1d4719b4-5d81-419d-b1b9-4ef00221c630)




# (Old below)

## Example:
![image](https://github.com/user-attachments/assets/3cf20101-55d2-4c1d-aba5-0f4633437b00)

## Its hard to make the centroids out:
- We can make them bigger, but this will make differentiating between them during inference much harder
![image](https://github.com/user-attachments/assets/e279d369-70fc-4e26-a9f3-32ec47edb0cb)
