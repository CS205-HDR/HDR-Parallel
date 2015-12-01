%matplotlib inline
import numpy as np
import scipy
import PIL
import matplotlib.pyplot as plt
import PIL.Image as im
from scipy import ndimage
from PIL import ImageEnhance

show = lambda img: plt.imshow(img)
gshow = lambda img: plt.imshow(img, cmap = plt.get_cmap('gray'))
rgb2gray = lambda rgb: np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

im = scipy.misc.imread('../test_large2.JPG',flatten=True)

def HE_serial(im_in):
    im = im_in.copy()
    histogram = [0]*256
    height, width = im.shape
    for i in im:
        for j in i:
            histogram[j] += 1
    histogram = np.array(histogram, dtype=float)/(width*height)
    cum_hist = np.cumsum(histogram)
    equal_hist = (cum_hist*256).astype(int)
    mapfunc = dict(zip(range(256), equal_hist))
    new_im = np.zeros_like(im)
    for i in range(height):
        for j in range(width):
            new_im[i,j] = mapfunc[im[i,j]]
    
    return new_im
    #plt.plot(range(0,256), cum_hist)

def histogram(im):
    histogram = [0]*256
    height, width = im.shape
    for i in im:
        for j in i:
            if j>255:
                j=255
            histogram[j] += 1
    histogram = np.array(histogram, dtype=float)/(width*height)
    plt.bar(range(256), histogram)
    return histogram


def histeq(im,nbr_bins=256):

    #get image histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)

    return im2.reshape(im.shape), cdf

if '__init__' == '__main__':
	im = scipy.misc.imread('../test_large2.JPG',flatten=True)
