
1. python packages needed before execution:
    1) PIL
    2) pyopencl
    3) numpy

2. Part 0: HDR Serial Implementation

    Program Folder/Files: 0-HDR-serial/HDR_serial.py

    For default testing images, you can run the program directly. If you would like to try different images set, replace the 'case'
    and 'cur_dir' values in line: "proj = hdr(case='3_lake', resize=1.0, img_type='jpg', cur_dir=r'../../testing_images')" under main function\
    where 'case' is the folder name where the images are stored and 'cur_dir' is the file path of 'case'.

3. Part 1 HDR Image Processing:
    Program Folders(5 steps of optimization): 1.1 xxx through 1.5 xxx
    Program Files:
        HDR.py: python driver for opencl
        hdr.cl: opencl implementation for parallel HDR image processing

    For default testing images, you can run the program directly. All the testing images are stored under '/testing_images' directory.
    If you want to test other images instead of the default 'lake', be sure to change all the file path in HDR.py file.

    Note: the execution may take several minutes because of loading large images.
    
Part 3 (Haosu)

Part 2 (Xinyan)