__kernel void
hdr(__global __read_only float *gpu_0,
           __global __read_only float *gpu_1,
           __global __read_only float *gpu_2,
           __global __read_only float *gpu_mask,
           __global __write_only float *gpu_out,
           __local float *buffer,
           int w, int h)
{
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            buffer[i * 3 + j] = gpu_mask[i * 3 + j];
        }
    }
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    int index = y * w + x;

    if(x >= w || y >= h)
        return;

    float temp = 0.0f;
    float weight = 0.0f;

    float r0 = gpu_0[y * w];
    float g0 = gpu_0[y * w + 1];
    float b0 = gpu_0[y * w + 2];

    float l0 = (min(min(r0, g0), b0) + max(max(r0, g0), b0)) * 0.5f;
    l0 /= 255.0f;
    float a0 = l0 - 0.5f;
    a0 = exp(-(a0 * a0)/1.0f);

    int curr0 = gpu_0[index] * a0;

    float r1 = gpu_1[y * w];
    float g1 = gpu_1[y * w + 1];
    float b1 = gpu_1[y * w + 2];

    float l1 = (min(min(r1, g1), b1) + max(max(r1, g1), b1)) * 0.5f;
    l1 /= 255.0f;
    float a1 = l1 - 0.5f;
    a1 = exp(-(a1 * a1)/1.0f);

    int curr1 = gpu_1[index] * a1;

    float r2 = gpu_2[y * w];
    float g2 = gpu_2[y * w + 1];
    float b2 = gpu_2[y * w + 2];

    float l2 = (min(min(r2, g2), b2) + max(max(r2, g2), b2)) * 0.5f;
    l2 /= 255.0f;
    float a2 = l2 - 0.5f;
    a2 = exp(-(a2 * a2)/1.0f);

    int curr2 = gpu_2[index] * a2;
    weight = a0 + a1 + a2;

    gpu_out[index] = (curr0 + curr1 + curr2) / weight;

    if ((x < w) && (y < h)) {

        gpu_out[w*y+x] = (gpu_0[w*y+x]+gpu_1[w*y+x]+gpu_2[w*y+x])/3;
    }

    //Saturation mask
    float tmp;
    if( (x < w) && (y <h)){
        tmp = 0.0f;
        for(int k = 0; k < 3; k++){
            tmp += gpu_out[y * w + k] * buffer[k * 3 + x];
        }
    }
    gpu_out[index] = tmp;

}
