#include <string>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "omp.h"
#include "processImage.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include <filesystem>
#include "estimation_utils.cuh"


__global__ void mseLoss(float* y_pred, float* y, float* resid_sq, int img_size) {
 int block_id  = blockIdx.x + blockIdx.y * gridDim.x;
 int i = block_id * (blockDim.x * blockDim.y)
         + (threadIdx.y * blockDim.x) + threadIdx.x;
 if (i < img_size) {
   float resid = y_pred[i] - y[i];
   if (resid < 0) {
     resid = resid * -1.;
   }
   resid_sq[i] = resid;
 }
 return;
}

__global__ void backPropagation(float* y_pred, float* y, float* img_padded, float* kernel, float* out, int kernel_radius, int width, int height, int p, int q) {
 int block_id  = blockIdx.x + blockIdx.y * gridDim.x;
 int jy = block_id * (blockDim.x * blockDim.y)
         + (threadIdx.y * blockDim.x) + threadIdx.x;
 int img_width_padded = width + 2*kernel_radius;
 if (jy < width*height) {
   int n_rows_down = (int) jy / width;
   int jx = jy + (img_width_padded * kernel_radius) + (n_rows_down*(2*kernel_radius)) + kernel_radius;
   int jx_col = jx % img_width_padded;
   out[jy] = 2*(y_pred[jy] - y[jy])*img_padded[jx + img_width_padded*(p-kernel_radius) + jx_col + q-kernel_radius];
 }
 return;
}


__host__ Image importImage(std::string file_path, int kernel_radius) {
 char* file_path_char = new char[file_path.length() + 1];
 strcpy(file_path_char, file_path.c_str());

 int width, height, channels;
 unsigned char *img = stbi_load(file_path_char, &width, &height, &channels, 3);

 int img_size = width * height;
 float* red_channel;
 float* green_channel;
 float* blue_channel;
 cudaMallocManaged(&red_channel,   img_size*sizeof(float));
 cudaMallocManaged(&green_channel, img_size*sizeof(float));
 cudaMallocManaged(&blue_channel,  img_size*sizeof(float));

 int position;
 #pragma omp parallel private(position)
 {
   #pragma omp for
   for (unsigned char *p = img; p < img + img_size * 3; p += 3) {
     position = (int) (p - img) / 3;
     red_channel[position]   = *p;
     green_channel[position] = *(p+1);
     blue_channel[position]  = *(p+2);
   }
 }

 int img_size_padded = (width + kernel_radius * 2) * (height + kernel_radius * 2);
 unsigned char *img_padded = (unsigned char*) malloc(img_size_padded * 3);
 processImage(img, img_padded, width, height, kernel_radius);

 float* red_channel_p;
 float* green_channel_p;
 float* blue_channel_p;
 cudaMallocManaged(&red_channel_p,   img_size_padded*sizeof(float));
 cudaMallocManaged(&green_channel_p, img_size_padded*sizeof(float));
 cudaMallocManaged(&blue_channel_p,  img_size_padded*sizeof(float));

 #pragma omp parallel private(position)
 {
   #pragma omp for
   for (unsigned char *p = img_padded; p < img_padded + img_size_padded * 3; p += 3) {
     position = (int) (p - img_padded) / 3;
     red_channel_p[position]   = *p;
     green_channel_p[position] = *(p+1);
     blue_channel_p[position]  = *(p+2);
   }
 }

 Image im;
 im.width = width;
 im.height = height;

 im.im_full = img;
 im.im_pad = img_padded;

 im.r = red_channel;
 im.g = green_channel;
 im.b = blue_channel;

 im.r_p = red_channel_p;
 im.g_p = green_channel_p;
 im.b_p = blue_channel_p;

 return im;
}
