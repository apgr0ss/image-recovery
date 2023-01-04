# image-recovery
Kernel estimation for high-resolution image recovery. The goal: given an arbitrarily-filtered image, reproduce the orginal. The code is GPU-compatible.

The project can be broken into two major steps:
1. Create the training data. Pass the original images through a filter and write the results to disk
2. Train the model on a subset of the training data. Evaluate on the dropped sample.

The model is a single convolution layer of dimension equal to the filter applied in step 1. Stochastic gradient descent is performed to minimize quadratic loss. Due to time constraints, the code is only compatible with images passed through a sharpen filter (though this can easily be substituted with another kernel).

This was a project for High Performance Computing for Engineering Applications (ME/CS/ECE 759) at University of Wisconsin-Madison. 
