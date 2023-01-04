#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "stdio.h"
#include "omp.h"
#include <random>
#include <chrono>
#include "stringSort.h"
#include "convolve2d.cuh"
#include "processImage.cuh"
#include "estimation_utils.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include <filesystem>

namespace fs = std::filesystem;
using std::chrono::system_clock;


int main(int argc, char *argv[]) {
  // File handling
  int n_threads = 20;
  omp_set_num_threads(n_threads);
  omp_set_nested(1);
  std::string path_orig_images = "data/orig";
  std::string path_filtered_images = "data/filtered/";
  std::vector<std::string> file_paths_orig;
  std::vector<std::string> file_paths_filtered;
  std::vector<std::string> file_names;

  for (const auto & entry : fs::directory_iterator(path_orig_images)) {
      file_paths_orig.push_back(entry.path());
      file_names.push_back(entry.path().filename());
  }

  for (const auto & entry : fs::directory_iterator(path_filtered_images)) {
      file_paths_filtered.push_back(entry.path());
  }

  mergeSort(file_names, 0, (int) file_names.size() - 1);
  mergeSort(file_paths_orig, 0, (int) file_paths_orig.size() - 1);
  mergeSort(file_paths_filtered, 0, (int) file_paths_filtered.size() - 1);


  // Build train and test sets
  int train_set_size = (int) file_names.size() * 0.7;


  std::vector<Image> X_test;
  std::vector<Image> y_test;
  int kernel_radius = 2;
  #pragma omp parallel for
  for (int i = train_set_size + 1; i < (int) file_names.size(); i++) {
    X_test.push_back(importImage(file_paths_filtered[i], kernel_radius));
    y_test.push_back(importImage(file_paths_orig[i], kernel_radius));
  }

  // Initialize kernel as gaussian noise with mean 0
  int kernel_dim = 2*kernel_radius + 1;
  float* kernel;
  cudaMallocManaged(&kernel,  kernel_dim*kernel_dim*sizeof(float));

  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed);
  std::normal_distribution<float> randNumbers(0, 0.001);
  for (int i = 0; i < kernel_dim*kernel_dim; i++) {
    kernel[i] = randNumbers(generator);
  }


  // Run training loop TODO: paralellize this (if time permits)
  int n_epochs = 3;
  float l_rate = 0.00001;
  float iter_loss = 0.0;
  float backprop_result_redux = 0.0;
  int abs_iter = 0;
  for (int i = 0; i < n_epochs; i++) {
    for (int j = 0; j < train_set_size; j++) {
      Image X_i = importImage(file_paths_filtered[j], kernel_radius);
      Image y_i = importImage(file_paths_orig[j], kernel_radius);

      int img_size = X_i.width * X_i.height;

      float* red_channel_pred;
      float* green_channel_pred;
      float* blue_channel_pred;

      cudaMallocManaged(&red_channel_pred,   img_size*sizeof(float));
      cudaMallocManaged(&green_channel_pred, img_size*sizeof(float));
      cudaMallocManaged(&blue_channel_pred,  img_size*sizeof(float));

      float* resid_sq_r;
      float* resid_sq_g;
      float* resid_sq_b;
      cudaMallocManaged(&resid_sq_r,  img_size*sizeof(float));
      cudaMallocManaged(&resid_sq_g,  img_size*sizeof(float));
      cudaMallocManaged(&resid_sq_b,  img_size*sizeof(float));

      float* backprop_result_vec_r;
      float* backprop_result_vec_g;
      float* backprop_result_vec_b;
      cudaMallocManaged(&backprop_result_vec_r,  img_size*sizeof(float));
      cudaMallocManaged(&backprop_result_vec_g,  img_size*sizeof(float));
      cudaMallocManaged(&backprop_result_vec_b,  img_size*sizeof(float));

      // Configure CUDA settings
      int block_width = 32;
      dim3 block_dim(block_width, block_width); // Covers 1024 pixels
      int grid_width  = (X_i.width + block_width - 1) / block_width;
      int grid_height = (X_i.height + block_width - 1) / block_width;
      dim3 grid_dim(grid_width, grid_height);

      // Run kernel over each channel
      convolve2d<<<grid_dim, block_dim>>>(X_i.r_p, red_channel_pred, kernel, X_i.width, X_i.height, kernel_radius);
      convolve2d<<<grid_dim, block_dim>>>(X_i.g_p, green_channel_pred, kernel, X_i.width, X_i.height, kernel_radius);
      convolve2d<<<grid_dim, block_dim>>>(X_i.b_p, blue_channel_pred, kernel, X_i.width, X_i.height, kernel_radius);

      cudaDeviceSynchronize();

      // Calculate loss here
      mseLoss<<<grid_dim, block_dim>>>(red_channel_pred, y_i.r, resid_sq_r, img_size);
      mseLoss<<<grid_dim, block_dim>>>(green_channel_pred, y_i.g, resid_sq_g, img_size);
      mseLoss<<<grid_dim, block_dim>>>(blue_channel_pred, y_i.b, resid_sq_b, img_size);

      cudaDeviceSynchronize();

      #pragma omp parallel for reduction(+:iter_loss)
      for (int s = 0; s < img_size; s++) {
        iter_loss += resid_sq_r[s] + resid_sq_g[s] + resid_sq_b[s];
      }
      if (j % 25 == 0 & j != 0) {
        printf("(Epoch %d, Iteration %d) loss %f...\n", i, j, (float) iter_loss / (img_size * 3));
      }
      iter_loss = 0.;
      // Get partial of derivative of loss with respect to kernel elements
      // Perform step of stochastic gradient descent on kernel weights
      for (int p = 0; p < kernel_dim; p++) {
        for (int q = 0; q < kernel_dim; q++) {
          backPropagation<<<grid_dim, block_dim>>>(red_channel_pred, y_i.r, X_i.r_p, kernel, backprop_result_vec_r,
                          kernel_radius, X_i.width, X_i.height, p, q);
          backPropagation<<<grid_dim, block_dim>>>(green_channel_pred, y_i.g, X_i.g_p, kernel, backprop_result_vec_g,
                          kernel_radius, X_i.width, X_i.height, p, q);
          backPropagation<<<grid_dim, block_dim>>>(blue_channel_pred, y_i.b, X_i.b_p, kernel, backprop_result_vec_b,
                          kernel_radius, X_i.width, X_i.height, p, q);
          cudaDeviceSynchronize();

          #pragma omp parallel for reduction(+ : backprop_result_redux)
          for (int d = 0; d < img_size; d++) {
            backprop_result_redux += (backprop_result_vec_r[d] + backprop_result_vec_g[d] + backprop_result_vec_b[d]);
          }

          backprop_result_redux = (float) backprop_result_redux / (img_size * 3);
          kernel[abs_iter] = kernel[abs_iter] - 2*l_rate*backprop_result_redux;
          backprop_result_redux = 0.;
          abs_iter += 1;
        }
      }
      abs_iter = 0;


      cudaFree(red_channel_pred);
      cudaFree(green_channel_pred);
      cudaFree(blue_channel_pred);

      cudaFree(resid_sq_r);
      cudaFree(resid_sq_g);
      cudaFree(resid_sq_b);

      cudaFree(backprop_result_vec_r);
      cudaFree(backprop_result_vec_g);
      cudaFree(backprop_result_vec_b);
    }
  }

  // Output estimated kernel
  std::ofstream out_file;
  out_file.open("kernel_est.txt");
  for (int i = 0; i < kernel_dim * kernel_dim - 1; i++) {
    out_file << kernel[i] << "\n";
  }
  out_file.close();
  cudaFree(kernel);

  return 0;
}
