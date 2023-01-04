#include "omp.h"
#include "stdio.h"
#include "processImage.cuh"



/*
processImage

Add border around image kernel_radius thick with 0s.

Credit to Mr Fooz:
https://stackoverflow.com/questions/12273047/most-efficient-method-to-pad-black-border-around-image-in-c
*/
__host__ void processImage(unsigned char* in, unsigned char* out, int img_width, int img_height, int kernel_radius) {
    // top border
    int top_or_bot_out_bytes = (img_width + kernel_radius * 2) * kernel_radius * 3;
    memset(out, 0, top_or_bot_out_bytes);

    // middle section
    unsigned char* in_row  = in;
    unsigned char* out_row = out + top_or_bot_out_bytes;
    for (int in_y = 0; in_y < img_height; in_y++) {
        // left border
        memset(out_row, 0, kernel_radius * 3);
        out_row += kernel_radius * 3;
        // center section
        memcpy(out_row, in_row, img_width * 3);
        out_row += img_width * 3;
        in_row  += img_width * 3;
        // right border
        memset(out_row, 0, kernel_radius * 3);
        out_row += kernel_radius * 3;
    }

    // bottom border
    memset(out, 0, top_or_bot_out_bytes);
  return;
}
