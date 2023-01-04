## Put SBATCH settings here

module load nvidia/cuda/11.6.0
nvcc generate_data.cu convolve2d.cu processImage.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -Xcompiler -fopenmp -o generateData
./generateData

nvcc estimate_kernel.cu convolve2d.cu processImage.cu stringSort.cpp estimation_utils.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -Xcompiler -fopenmp -o estimateKernel
./estimateKernel