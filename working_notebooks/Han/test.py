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

program = cl.Program(context, open('HDR_mask.cl').read()).build(options='')


im0 = scipy.misc.imread('test_large.jpg', flatten=True)
him0 = im0.copy()
him0 = np.array(him0, dtype=np.float32)
him0.shape
mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype(np.float32)

print mask
out = np.zeros_like(him0).astype(np.float32)
gpu_0 = cl.Buffer(context, cl.mem_flags.READ_ONLY, him0.size * 4)
gpu_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, out.size * 4)
# gpu of mask
gpu_mask = cl.Buffer(context, cl.mem_flags.READ_ONLY, mask.size * 4)

cl.enqueue_copy(queue, gpu_0, him0, is_blocking=False)
cl.enqueue_copy(queue, gpu_mask, mask, is_blocking=False)
#cl.enqueue_copy(queue, gpu_sat, saturation, is_blocking=False)


local_size = (8, 8)  # 64 pixels per work group
global_size = tuple([round_up(g, l) for g, l in zip(him0.shape[::-1], local_size)])

width = np.int32(him0.shape[1]) # 3
height = np.int32(him0.shape[0]) # 499392
halo = np.int32(1)

# Set up a (N+2 x N+2) local memory buffer.
# +2 for 1-pixel halo on all sides, 4 bytes for float.
buf_size = (np.int32(local_size[0] + 2 * halo), np.int32(local_size[1] + 2 * halo))
buf_w = np.int32(local_size[0] + 2)
buf_h = np.int32(local_size[1] + 2)
local_memory = cl.LocalMemory(4 * width*height)
#event = program.mask(queue, global_size, local_size,
                           gpu_0, gpu_mask, gpu_out, local_memory,
                           buf_size[0], buf_size[1], width, height, halo)

event = program.mask_nobuffer(queue, global_size, local_size,
                           gpu_0, gpu_mask, gpu_out,
                           width, height, np.int32(mask.shape[1]), np.int32(mask.shape[0]))

cl.enqueue_copy(queue, out, gpu_out, is_blocking=True)

seconds = (event.profile.end - event.profile.start) / 1e9
