## Usage
```python
%pip install git+https://github.com/helalabcd/hela_datasets
```

```python
from hela_datasets import LocationDirectionMapDatasetds = LocationDirectionMapDataset("miniHela/train")

for burst in ds:
    x, y = burst
```

## Example:
![image](https://github.com/user-attachments/assets/3cf20101-55d2-4c1d-aba5-0f4633437b00)

## Its hard to make the centroids out:
- We can make them bigger, but this will make differentiating between them during inference much harder
![image](https://github.com/user-attachments/assets/e279d369-70fc-4e26-a9f3-32ec47edb0cb)
