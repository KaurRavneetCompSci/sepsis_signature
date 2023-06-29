import numpy as np
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
from tsaug.visualization import plot
import matplotlib.pyplot as plt

print("---------tsaug Loaded----------")

my_augmenter = (TimeWarp() * 1  # random time warping 5 times in parallel
     + Crop(size=300)  # random crop subsequences with length 300
     + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
     + Drift(max_drift=(0.1, 0.5)) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
     + Reverse() @ 0.5  # with 50% probability, reverse the sequence
)

print("---------Data being loaded----------")

X = np.load("/Users/harpreetsingh/tsaug/tsaug/docs/notebook/X.npy")
Y = np.load("/Users/harpreetsingh/tsaug/tsaug/docs/notebook/Y.npy")
plot(X, Y)

print("---------Data being augmented----------")

X_aug, Y_aug = my_augmenter.augment(X, Y)
plot(X_aug, Y_aug)
plt.show()