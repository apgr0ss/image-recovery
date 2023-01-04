#include <string>
#include "stdio.h"
#include "omp.h"
#include <chrono>
#include "convolve2d.cuh"
#include "processImage.cuh"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <vector>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

using std::chrono::system_clock;

/*
  Main file to pass in original images and generate
  filtered versions using true kernel
*/

int main(int argc, char *argv[]) {
  int n_threads = 20;
  omp_set_num_threads(n_threads);
  omp_set_nested(1);
  std::string path = "data/orig/";
  std::vector<std::string> file_paths;
  std::vector<std::string> file_names;
  for (const auto & entry : fs::directory_iterator(path)) {
      file_paths.push_back(entry.path());
      file_names.push_back(entry.path().filename());
  }

  #pragma omp parallel for
  for (int i = 0; i < (int) file_paths.size(); i++) {
    std::string file_path = file_paths[i];
    char* file_path_char = new char[file_path.length() + 1];
    strcpy(file_path_char, file_path.c_str());
    // Import original image
    int width, height, channels;
    unsigned char *img = stbi_load(file_path_char, &width, &height, &channels, 3);

    // Original channels
    float* red_channel;
    float* green_channel;
    float* blue_channel;

    // Filtered channels
    float* red_channel_f;
    float* green_channel_f;
    float* blue_channel_f;

    // Kernel
    float* kernel_ptr;

    // Define static kernel (sharpen)
    float kernel[25] = {0., 0., -1., 0., 0., 0., 0., -1., 0., 0., -1., -1., 5., -1., -1., 0., 0., -1., 0., 0., 0., 0., -1., 0., 0.};

    int kernel_radius = 2;

    int img_size = width * height;
    int img_size_padded = (width + kernel_radius * 2) * (height + kernel_radius * 2);

    // Allocate memory to device & host for original channels
    cudaMallocManaged(&red_channel,   img_size_padded*sizeof(float));
    cudaMallocManaged(&green_channel, img_size_padded*sizeof(float));
    cudaMallocManaged(&blue_channel,  img_size_padded*sizeof(float));

    // Allocate memory to device & host for filtered channels
    cudaMallocManaged(&red_channel_f, img_size*sizeof(float));
    cudaMallocManaged(&green_channel_f, img_size*sizeof(float));
    cudaMallocManaged(&blue_channel_f, img_size*sizeof(float));

    // Allocate memory to device & host for kernel
    cudaMallocManaged(&kernel_ptr, 25*sizeof(float));

    // Pad image kernel_radius thick with 0s
    unsigned char *img_padded = (unsigned char*) malloc(img_size_padded * 3);
    processImage(img, img_padded, width, height, kernel_radius);


    // Fill in kernel
    int n_cells_kernel = 25;
    for (int j = 0; j < n_cells_kernel; j++) {
      kernel_ptr[j] = kernel[j];
    }

    // Transform raw input to arrays
    // Each channel is represented in row-major form
    int position;
    #pragma omp parallel private(position)
    {
      #pragma omp for
      for (unsigned char *p = img_padded; p < img_padded + img_size_padded * 3; p += 3) {
        position = (int) (p - img_padded) / 3;
        red_channel[position]   = (float) *p;
        green_channel[position] = (float) *(p+1);
        blue_channel[position]  = (float) *(p+2);
      }
    }


    int block_width = 32;
    dim3 block_dim(block_width, block_width); // Covers 1024 pixels

    int padded_width = width + 2*kernel_radius;
    int padded_height = height + 2*kernel_radius;
    int grid_width  = (padded_width + block_width - 1) / block_width;
    int grid_height = (padded_height + block_width - 1) / block_width;
    dim3 grid_dim(grid_width, grid_height);


    // TODO: implement shared memory
    // Determine how much space should be allocated for shared memory
    // int padded_width = block_width + 2*kernel_radius;
    // int shared_mem_size = sizeof(int) * padded_width * padded_width;

    // Run kernel over each channel
    convolve2d<<<grid_dim, block_dim>>>(red_channel, red_channel_f, kernel_ptr, width, height, kernel_radius);
    convolve2d<<<grid_dim, block_dim>>>(green_channel, green_channel_f, kernel_ptr, width, height, kernel_radius);
    convolve2d<<<grid_dim, block_dim>>>(blue_channel, blue_channel_f, kernel_ptr, width, height, kernel_radius);

    cudaDeviceSynchronize();

    uint8_t* output_image = new uint8_t[width*height*3];
    for (int k = 0; k < width*height*3; k += 3) {
      int position = (int) k / 3;
      output_image[k] =   (uint8_t) red_channel_f[position];
      output_image[k+1] = (uint8_t) green_channel_f[position];
      output_image[k+2] = (uint8_t) blue_channel_f[position];
    }


    std::string path = "data/filtered/";
    std::string name = file_names[i];
    std::string full_name = path + name;
    char* file_out_name = new char[full_name.length() + 1];
    strcpy(file_out_name, full_name.c_str());
    stbi_write_png(file_out_name, width, height, 3, output_image, width * sizeof(uint8_t) * 3);

    // Free memory
    cudaFree(red_channel);
    cudaFree(green_channel);
    cudaFree(blue_channel);

    cudaFree(red_channel_f);
    cudaFree(green_channel_f);
    cudaFree(blue_channel_f);

    cudaFree(kernel_ptr);
    delete[] output_image;
  }
  return 0;
}
