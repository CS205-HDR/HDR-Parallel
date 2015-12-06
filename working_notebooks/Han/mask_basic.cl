
// This is the non-buffer cl for mask


// This is the mask calculation version


float get_right_pix(__global __read_only float *labels,
           int w, int h,
           int x, int y);

// for given point in buffer, make sure it is within bounds of workgroup

//float get_right_pix(__global __read_only float *gpu_in,
//           int w, int h,
//           int curr_x, int curr_y) {
//    if (curr_x < 0) {
//        curr_x = 0;
//    }
//    if (curr_x > w-1) {
//        curr_x = w-1;
//    }
//    if (curr_y < 0) {
//        curr_y = 0;
//    }
//    if (curr_y > h-1) {
//        curr_y = h-1;
//    }
//    return gpu_in[curr_y * w + curr_x];
//}

float
get_right_pix(__global __read_only float *gpu_in,
                  int w, int h,
                  int x, int y)
{
    if ((x < 0) || (x >= w) || (y < 0) || (y >= h))
        return w * h;
    return gpu_in[y * w + x];
}





__kernel void
mask_nobuffer(__global __read_only float *gpu_in,
                 __global __read_only float *gpu_mask,
                 __global __write_only float *gpu_out,
                 int w, int h,
                 int mask_w, int mask_h)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if((y < h-1)  && (x < w-1) && (x>0) && (y>0)){
        float out_val = 0;
        for(int mh = 0; mh < mask_h; mh++)
            for(int mw = 0; mw < mask_w; mw++)
                out_val += gpu_mask[mh * mask_w + mw] * gpu_in[(y - mask_h / 2 + mh) * w + x - mask_w / 2 + mw];


        //printf("%f", out_val);

        gpu_out[x + w * y] = out_val;

    }
}

