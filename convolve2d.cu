#include "stdio.h"
#include "convolve2d.cuh"
#include "cuda.h"
#include "cuda_runtime.h"

/*
convolve2d

Convolve an RGB image with a given filter
*/

__global__ void convolve2d(float *img,
                           float *output,
                           float *kernel,
                           int img_width,
                           int img_height,
                           int kernel_radius) {
    /*
    TODO - implement shared memory solution
    // Define shared memory
    extern __shared__ float s[];
    int *input_shared  = (int*)s;
    */

    int block_id  = blockIdx.x + blockIdx.y * gridDim.x;
    int global_thread_id = block_id * (blockDim.x * blockDim.y)
                            + (threadIdx.y * blockDim.x) + threadIdx.x;


    int img_width_padded = img_width + 2*kernel_radius;


    // Create conditions to identify threads that map to pixels in actual image
    // Only want to process these threads
    int top_condition = (int) global_thread_id >= img_width_padded * kernel_radius;
    int bot_condition = (int) global_thread_id < img_width_padded * (kernel_radius + img_height);

    int n_rows_down = (int) (global_thread_id / img_width_padded);
    int left_cond  = (int) (global_thread_id - (n_rows_down * img_width_padded)) > kernel_radius - 1;
    int right_cond = (int) (global_thread_id - (n_rows_down * img_width_padded)) < kernel_radius + img_width;
    if (top_condition & bot_condition & left_cond & right_cond) {
        int sum = 0;
        int kernel_iter = 0;
        for (int i = -1*kernel_radius;  i < kernel_radius + 1; i++) {
            for (int j = -1*kernel_radius; j < kernel_radius + 1; j++) {
                sum += img[global_thread_id + img_width_padded * i + j] * kernel[kernel_iter];
                kernel_iter += 1;
            }
        }
        output[global_thread_id - (img_width_padded * kernel_radius) - ((n_rows_down - kernel_radius)*(2*kernel_radius)) - kernel_radius] = sum;
    }
    return;
}
