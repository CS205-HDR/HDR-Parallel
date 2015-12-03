__author__ = 'haosutang'

import pyopencl as cl
import numpy as np
import pylab

import scipy
import PIL
import PIL.Image as im
from scipy import ndimage
from PIL import ImageEnhance
import random
import os

from AHE import *

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r


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
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open('ACH_2.cl').read()).build(options='')


    # Read in images, im_orig is a large image, im is a small one.
    im_orig = scipy.misc.imread('../test_large2.JPG',flatten=True)
    # Use a small image to test (by 12*12)
    im = np.array([im_orig[i][::12] for i in range(len(im_orig)) if i%12==0])


    #################################
    # 1. HE_serial on large img
    #################################



    AHE_out = np.zeros_like(im)

    gpu_im = cl.Buffer(context, cl.mem_flags.READ_ONLY, im.size * 4)
    gpu_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, AHE_out.size * 4)

    local_size = (8, 8)  # 64 pixels per work group
    global_size = tuple([round_up(g, l) for g, l in zip(im.shape[::-1], local_size)])


    pad_size = np.int32(4)
    # Set up a (N+8 x N+8) local memory buffer.
    # +2 for 1-pixel halo on all sides, 4 bytes for float.
    local_memory = cl.LocalMemory(4 * (local_size[0] + pad_size) * (local_size[1] + pad_size))
    # Each work group will have its own private buffer.
    buf_width = np.int32(local_size[0] + 2*pad_size)
    buf_height = np.int32(local_size[1] + 2*pad_size)
    halo = np.int32(pad_size)



    print im.shape
    width = np.int32(im.shape[1])
    height = np.int32(im.shape[0])

    #max_iters = np.int32(1024)

    cl.enqueue_copy(queue, gpu_im, im, is_blocking=False)

    event = program.AHE_test(queue, global_size, local_size,
                           gpu_im, gpu_out, local_memory,
                           width, height,
                           buf_width, buf_height, halo,2*pad_size+1)

    cl.enqueue_copy(queue, AHE_out, gpu_out, is_blocking=True)

    seconds = (event.profile.end - event.profile.start) / 1e9

    gshow(AHE_out.astype(int))
    print "------------------------------------------"
    print "AHE serial on small image with size: ", im.shape
    print "Used time: ", seconds
    print "------------------------------------------"

    # id_comp2 = np.reshape(out, (612,816,3)).astype(np.uint8)
    # print 'shape', id_comp2.shape
    # print id_comp2[:20]
    # im_comp = Image.fromarray(id_comp2, 'RGB')
    #
    # print 'shape', id_comp2.shape
    # print id_comp2[:20]
    # im_comp.show()