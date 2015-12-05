
from __future__ import division
import pyopencl as cl
import numpy as np
import pylab
from PIL import Image

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

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open('hdr.cl').read()).build(options='')

    #in_coords, out_counts = make_coords()
    #real_coords = np.real(in_coords).copy()
    #imag_coords = np.imag(in_coords).copy()

    im0 = np.array(Image.open("orig_0.jpg").getdata())
    him0 = im0.astype(np.float32).copy()

    im1 = np.array(Image.open("orig_1.jpg").getdata())
    him1 = im1.astype(np.float32).copy()
    im2 = np.array(Image.open("orig_2.jpg").getdata())
    him2 = im2.astype(np.float32).copy()
    im3 = np.array(Image.open("orig_3.jpg").getdata())
    him3 = im3.astype(np.float32).copy()

    out = np.empty_like(him0)

    gpu_0 = cl.Buffer(context, cl.mem_flags.READ_ONLY, him0.size * 4)
    gpu_1 = cl.Buffer(context, cl.mem_flags.READ_ONLY, him1.size * 4)
    gpu_2 = cl.Buffer(context, cl.mem_flags.READ_ONLY, him2.size * 4)
    gpu_3 = cl.Buffer(context, cl.mem_flags.READ_ONLY, him3.size * 4)
    gpu_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, him0.size * 4)

    local_size = (8, 8)  # 64 pixels per work group
    global_size = tuple([round_up(g, l) for g, l in zip(him0.shape[::-1], local_size)])

    print him0.shape
    width = np.int32(him0.shape[1])
    height = np.int32(him0.shape[0])

    #max_iters = np.int32(1024)

    cl.enqueue_copy(queue, gpu_0, him0, is_blocking=False)
    cl.enqueue_copy(queue, gpu_1, him1, is_blocking=False)
    cl.enqueue_copy(queue, gpu_2, him2, is_blocking=False)
    cl.enqueue_copy(queue, gpu_3, him3, is_blocking=False)

    event = program.hdr(queue, global_size, local_size,
                               gpu_0, gpu_1, gpu_2, gpu_3, gpu_out,
                               width, height)

    cl.enqueue_copy(queue, out, gpu_out, is_blocking=True)

    seconds = (event.profile.end - event.profile.start) / 1e9
    print("{} Million Complex FMAs in {} seconds, {} million Complex FMAs / second".format(out.sum() / 1e6, seconds, (out.sum() / seconds) / 1e6))


    id_comp2 = np.reshape(out, (612,816,3)).astype(np.uint8)
    print 'shape', id_comp2.shape
    print id_comp2[:20]
    im_comp = Image.fromarray(id_comp2, 'RGB')

    print 'shape', id_comp2.shape
    print id_comp2[:20]
    im_comp.show()