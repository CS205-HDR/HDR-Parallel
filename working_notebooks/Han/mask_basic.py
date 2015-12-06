
# This is the basic non-buffer version of mask

from __future__ import division
import pyopencl as cl
import numpy as np
from PIL import Image
import pylab
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage


def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r


if __name__ == '__main__':
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

    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open('mask_basic.cl').read()).build(options='')


    im0 = scipy.misc.imread('pic.jpg', flatten=True)
    him0 = im0.copy()
    him0 = np.array(him0, dtype=np.float32)

    print him0.shape

    # Original
    #mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype(np.float32)
    # sharpen
    #mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)
    # Box blur
    mask = (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.float32)
    # Edge detection
    #mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).astype(np.float32)
    # Edge detection2
    #mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).astype(np.float32)
    # Gaussian blur
    #mask = (1/16)*np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).astype(np.float32)

    #print mask
    out = np.zeros_like(him0).astype(np.float32)
    gpu_0 = cl.Buffer(context, cl.mem_flags.READ_ONLY, him0.size * 4)
    gpu_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, out.size * 4)
    # gpu of mask
    gpu_mask = cl.Buffer(context, cl.mem_flags.READ_ONLY, mask.size * 4)

    cl.enqueue_copy(queue, gpu_0, him0, is_blocking=False)
    cl.enqueue_copy(queue, gpu_mask, mask, is_blocking=False)


    local_size = (8, 8)  # 64 pixels per work group
    global_size = tuple([round_up(g, l) for g, l in zip(him0.shape[::-1], local_size)])

    width = np.int32(him0.shape[1]) # 3
    height = np.int32(him0.shape[0]) # 499392


    event = program.mask_nobuffer(queue, global_size, local_size,
                               gpu_0, gpu_mask, gpu_out,
                               width, height, np.int32(mask.shape[1]), np.int32(mask.shape[0]))

    cl.enqueue_copy(queue, out, gpu_out, is_blocking=True)

    seconds = (event.profile.end - event.profile.start) / 1e9
    print 'Seconds: ', seconds


    #print out
    pylab.imshow(out, cmap=plt.cm.gray)
    pylab.show()
