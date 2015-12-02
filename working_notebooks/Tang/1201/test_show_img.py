import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

plt.interactive(False)

im_orig = scipy.misc.imread('../test_large2.JPG',flatten=True)
# Use a small image to test (by 12*12)
im = np.array([im_orig[i][::12] for i in range(len(im_orig)) if i%12==0])

plt.imshow(im, cmap=plt.get_cmap('gray'))
plt.show()