__kernel void
mandelbrot(__global __read_only float *id0,
           __global __read_only float *id1,
           __global __read_only float *id2,
           __global __read_only float *id3,
           __global __write_only int *hdr_out,
           int w, int h)
{
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

//    float c_real, c_imag;
//    float z_real, z_imag;
//    int iter;
//    float z_real_temp;

    if ((x < w) && (y < h)) {
        // YOUR CODE HERE
//        c_real = coords_real[x+y*w];
//        c_imag = coords_imag[x+y*w];
//        z_real = 0;
//        z_imag = 0;

//        iter = 0;
//        while(z_real * z_real + z_imag * z_imag < 4 && iter < max_iter){
//            z_real_temp = z_real;
//            z_real = z_real * z_real - z_imag * z_imag + c_real;
//            z_imag = z_imag * z_real_temp * 2 + c_imag;
//            iter++;
//        }

        hdr_out[x + y * w] = (id0[x+y*w] + id1[x+y*w] + id2[x+y*w] + id3[x+y*w])/4;
    }
}
