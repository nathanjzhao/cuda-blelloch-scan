#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <cuda_runtime.h>

__global__ void blelloch_scan_kernel(int* d_out, int* d_in, int n) {
    extern __shared__ int temp[];

    // Thread ID
    int thid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    temp[2*thid] = (2*thid < n) ? d_in[2*thid] : 0;
    temp[2*thid+1] = (2*thid+1 < n) ? d_in[2*thid+1] : 0;

    // Upsweep phase
    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (thid == 0) {
        temp[n - 1] = 0;
    }

    // Downsweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // Write results to device memory
    if (2*thid < n) d_out[2*thid] = temp[2*thid];
    if (2*thid+1 < n) d_out[2*thid+1] = temp[2*thid+1];
}

void gpu_blelloch_scan(int* h_out, int* h_in, int n) {
    int *d_in, *d_out;
    
    // Allocate device memory
    cudaMalloc((void**)&d_in, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));

    // Copy input from host to device memory
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);

    // Compute launch configuration -- will have less blocks with more elements in array
    int block_size = 128;
    int num_blocks = (n + block_size - 1) / block_size;

    // Launch kernel
    blelloch_scan_kernel<<<num_blocks, block_size, n * sizeof(int)>>>(d_out, d_in, n);

    // Copy result from device to host memory
    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
}

void cpu_naive_sum(int* h_out, int* h_in, int n) {
    h_out[0] = 0;
    for (int i = 1; i < n; i++) {
        h_out[i] = h_out[i - 1] + h_in[i-1];
    }
}

void cpu_blelloch_scan(int* h_out, int* h_in, int n) {

    // Perform upsweep
    // We execute our operator (here, addition) on array elements 1 apart, then 2, etc.
    // until we converge at the end of the array.
    for (int d = 1; d < n; d *= 2) {
        for (int i = 0; i < n; i += 2 * d) {
            h_in[i + 2 * d - 1] += h_in[i + d - 1];
        }
    }

    // Set last element to 0
    h_in[n - 1] = 0;

    // Perform downsweep
    // We branch out from the *last* element, executing a downsweep iteration on
    // elements n/2 apart, then n/4, etc. in an inverted tree until we have filled all array indices.
    // We carry down L+R to the right child, and set the left child to R (for L and R as left/right parents)
    for (int d = n / 2; d > 0; d /= 2) {
        for (int i = 0; i < n; i += 2 * d) {
            int t = h_in[i + d - 1];
            h_in[i + d - 1] = h_in[i + 2 * d - 1];
            h_in[i + 2 * d - 1] += t;
        }
    }

    // Copy result to output array
    for (int i = 0; i < n; i++) {
        h_out[i] = h_in[i];
    }
}