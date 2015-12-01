from __future__ import division
import pyopencl as cl
import numpy as np
import pylab
import matplotlib.pyplot as plt

import scipy
import PIL
import PIL.Image as im
from scipy import ndimage
from PIL import ImageEnhance
import random

from AHE import *



def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

# create coordinates, along with output count array
def make_coords(center=(-0.575 - 0.575j),
                width=0.0025,
                count=4000):

    x = np.linspace(start=(-width / 2), stop=(width / 2), num=count)
    xx = center + (x + 1j * x[:, np.newaxis]).astype(np.complex64)
    return xx, np.zeros_like(xx, dtype=np.uint32)


if __name__ == '__main__':
    #################################
    # Setting up environment
    #################################
    # List our platforms
    platforms = cl.get_platforms()
    print 'The platforms detected are:'
    print '---------------------------'
    for platform in platforms:
        print platform.name, platform.vendor, 'version:', platform.version

    # List devices in each platform
    for platform in platforms:
        print 'The devices detected on platform', platform.name, 'are:'
        print '---------------------------'
        for device in platform.get_devices():
            print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
            print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
            print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
            print 'Maximum work group size', device.max_work_group_size
            print '---------------------------'

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices)
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.

    
    #################################
    # Read in image
    #################################
    #Helper functions
    show = lambda img: plt.imshow(img)
    gshow = lambda img: plt.imshow(img, cmap = plt.get_cmap('gray'))
    rgb2gray = lambda rgb: np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

    im_orig = scipy.misc.imread('../test_large2.JPG',flatten=True)
    # Use a small image to test (by 12*12)
    im = np.array([im_orig[i][::12] for i in range(len(im_orig)) if i%12==0])
    AHE_out = np.zeros_like(im, dtype=np.float)

    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open('AHE.cl').read()).build(options='')

    gpu_id = cl.Buffer(context, cl.mem_flags.READ_ONLY, im.size * 4)

    gpu_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, AHE_out.size * 4)

    local_size = (8, 8)  # 64 pixels per work group
    global_size = (id0.shape[0], id0.shape[1])
    width = np.int32(id0.shape[1])
    height = np.int32(id1.shape[0])
    gpu_real = cl.Buffer(context, cl.mem_flags.READ_ONLY, real_coords.size * 4)

    cl.enqueue_copy(queue, gpu_id, im, is_blocking=False)




    event = program.AHE(queue, global_size, local_size,
                        gpu_id,
                        width, height)

    cl.enqueue_copy(queue, gpu_out, hdr_out, is_blocking=True)

    seconds = (event.profile.end - event.profile.start) / 1e9
    print("{} seconds".format(seconds))
    pylab.imshow(hdr_out)
    pylab.show()
