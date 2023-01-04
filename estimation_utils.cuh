#ifndef ESTIMATION_UTILS_CUH
#define ESTIMATION_UTILS_CUH

struct Image {
  int width;
  int height;

  unsigned char* im_full;
  unsigned char* im_pad;

  float* r;   // red channel
  float* g;   // green channel
  float* b;   // blue channel

  float* r_p; // red channel (padded)
  float* g_p; // green channel (padded)
  float* b_p; // blue channel (padded)
};

__global__ void mseLoss(float* y_pred, float* y, float* resid_sq, int img_size);

__global__ void backPropagation(float* y_pred, float* y, float* img_padded, float* kernel, float* out, int kernel_radius, int width, int height, int p, int q);

__host__ Image importImage(std::string file_path, int kernel_radius);

#endif
