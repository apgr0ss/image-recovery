#ifndef CONVOLVE2D_CUH
#define CONVOLVE2D_CUH


/*
convolve2d

Convolve an RGB image with a given filter

General steps:
1.For a given block, initialize shared memory, where amount of memory upper bounded by size<int> times
  number of elements contained in block:
      top left pixel  + kernelWidth up and left
      top right pixel + kernelWidth up and right
      bot left pixel  + kernelWidth down and left
      bot right pixel + kernelWidth down and right

2. For each thread, populate shared memory with value in its position. If thread is on image boundary, populate OOB with 0s.
3. Syncthreads
4. For each thread perform elementwise multiplication of entries in shared memory
   within kernelWidth of thread's representative pixel with corresponding kernel
   element, sum the result and store in respective position in output array
*/

__global__ void convolve2d(float *img,
                           float *output,
                           float *kernel,
                           int img_width,
                           int img_height,
                           int kernel_radius);
#endif
