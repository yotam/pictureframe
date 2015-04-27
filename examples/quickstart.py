import numpy as np
from numpy.random import random_sample
from skimage import img_as_float
from skimage.io import imread

from pictureframe import PictureFrame

image = img_as_float(imread('sample_data/image.png'))
depth = img_as_float(imread('sample_data/depth.png'))

# CONSTRUCTORS

# dict of array name to array
pf = PictureFrame({"image": image, "depth": depth})

# quick look at the the data
print(pf)

# INDEXING

# we pass the indexing arguments down to numpy, so everything
# works as you would expect (apart from maybe broadcasting)

# slicing returns a new PictureFrame with views of original data
print(pf[30:, 80:220])

# cartesian indexing returns a new PictureFrame with copies of the data
# note the new shape
print(pf[[1, 3, 4], [40, 10, 11]])

# boolean or mask indexing works fine too
mask = np.random.random_sample(depth.shape) > 0.5
print(pf[mask])

# CONVENIENCE FUNCTIONS

# zoom function returns a new PictureFrame with resized data arrays
# order of interpolation is by default 2 for float arrays and 0 for
# integer arrays, but this can be overridden
print(pf.zoom(0.5))

# pandas/SQL-like groupby function iterates over sub-PictureFrames
# corresponding to each label value
# here we use an "external" array...
labels = imread('sample_data/labels.png')

for label, pf_group in pf.groupby(labels):
    print(label, pf_group)

# however we can add it to the PictureFrame
pf['labels'] = labels

# and group by the name of the array
for k, pf_group in pf.groupby('labels'):
    print(k, pf_group)

# browse function lets you see all array data with matplotlib
pf.browse()

# ASSIGNMENT

indices = np.array([10, 7, 3, 0, 12])

# copy some data to a new PictureFrame and alter the values
other_pf = pf[indices]
other_pf['image'] = random_sample(other_pf.image.shape)

# assignment of values between corresponding arrays handled internally
pf[indices] = other_pf
