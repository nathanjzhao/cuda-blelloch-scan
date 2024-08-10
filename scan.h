#ifndef SCAN_H__
#define SCAN_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for Blelloch scan
__global__ void blelloch_scan_kernel(int* d_out, int* d_in, int n);

// GPU implementation of Blelloch scan
void gpu_blelloch_scan(int* h_out, int* h_in, int n);

// CPU implementation of naive sum scan
void cpu_naive_sum(int* h_out, int* h_in, int n);

// CPU implementation of Blelloch scan
void cpu_blelloch_scan(int* h_out, int* h_in, int n);

#endif // SCAN_H__