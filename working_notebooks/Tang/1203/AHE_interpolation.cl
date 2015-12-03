// Get histogram in every segmentation
__kernel void
AHE_interpolation_hist(__global __read_only float *in_values,
           __global __write_only int *histmapping,
           int w, int h,
           int seg_width, int seg_height){

    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Number of the segmentations in the width and height
    const int w_seg = w/seg_width;
    const int h_seg = h/seg_height;

    if((y<h) && (x<w)){
        float currentpixel = in_values[y*w+x];
        // Atomic add to the histogram
        atomic_add(&histmapping[256*(int)(x/seg_width)+(int)(y/seg_height)*w_seg*256+(int)(currentpixel)],1);
    }
}

// Transformation through bilinear interpolation
__kernel void
AHE_interpolation_transform(__global __read_only float *in_values,
           __global __read_only int *cumhist,
           __global __write_only float *out_values,
           int w, int h,
           int seg_width, int seg_height){

    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if((y<h) && (x<w)){
        const int w_seg = w/seg_width;
        const int h_seg = h/seg_height;

        int left = (x+seg_width/2)/seg_width;
        int right = left+1;
        int up = (y+seg_height/2)/seg_height;
        int down = up+1;

        float currentpixel = in_values[y*w+x];


        // tile coordinates
        int tile1_x = left-1;
        int tile1_y = up-1;
        int tile2_x = left;
        int tile2_y = up-1;
        int tile3_x = left-1;
        int tile3_y = up;
        int tile4_x = left;
        int tile4_y = up;

        // Mapped values at the four neighboring nodes
        int map1 = cumhist[tile1_y*256*w_seg + tile1_x*256 + (int)currentpixel];
        int map2 = cumhist[tile2_y*256*w_seg + tile2_x*256 + (int)currentpixel];
        int map3 = cumhist[tile3_y*256*w_seg + tile3_x*256 + (int)currentpixel];
        int map4 = cumhist[tile4_y*256*w_seg + tile4_x*256 + (int)currentpixel];

        // Bilinear interpolation coefficients
        float a = (y-((float)tile1_y+0.5)*seg_height)/seg_height;
        float b = (x-((float)tile1_x+0.5)*seg_width)/seg_width;

        if (a<0){   // When going out of the upper bound
            a=0;
        }
        if (a>1){   // When going out of the lower bound
            a=1;
        }
        if (b<0){   // When going out of the left bound
            b=0;
        }
        if (b>1){   // When going out of the right bound
            b=1;
        }

        // Bilinear interpolation
        out_values[y*w+x] = (1-a)*((1-b)*map1+b*map2)+a*((1-b)*map3+b*map4);


    }

}