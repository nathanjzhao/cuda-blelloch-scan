#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <cuda_runtime.h>

__global__ void blelloch_scan_kernel(int* d_out, int* d_in, int n) {
    extern __shared__ int temp[];

    int thid = threadIdx.x;
    int block_size = blockDim.x;
    int block_offset = blockIdx.x * block_size * 2;
    int offset = 1;

    // Load input into shared memory
    int ai = block_offset + thid;
    int bi = ai + block_size;
    temp[thid] = (ai < n) ? d_in[ai] : 0;
    temp[thid + block_size] = (bi < n) ? d_in[bi] : 0;

    // Upsweep phase
    for (int d = block_size; d > 0; d >>= 1) {
        __syncthreads();

        // Parallelizes d threads at same time
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            if (bi < 2 * block_size) temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (thid == 0) temp[2 * block_size - 1] = 0;

    // Downsweep phase
    for (int d = 1; d < 2 * block_size; d *= 2) {
        offset >>= 1;
        __syncthreads();

        // Parallelizes d threads at same time
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            if (bi < 2 * block_size) {
                int t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
    }

    __syncthreads();

    // Write results to device memory
    if (ai < n) d_out[ai] = temp[thid];
    if (bi < n) d_out[bi] = temp[thid + block_size];
}

// Extract block sums 
__global__ void extract_block_sums(int* d_block_sums, int* d_out, int n, int block_size) {
    int idx = threadIdx.x;
    int block_end = (idx + 1) * block_size - 1;
    if (block_end < n) {
        d_block_sums[idx] = d_out[block_end];
    } else if (idx * block_size < n) {
        d_block_sums[idx] = d_out[n - 1];
    } else {
        d_block_sums[idx] = 0;
    }
}

// Add block sums of previous blocks to each item in the block
__global__ void add_block_sums(int* d_out, int* d_block_sums, int n, int block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = blockIdx.x;
    
    if (idx < n && block_idx > 0) {
        d_out[idx] += d_block_sums[block_idx - 1];
    }
}

void gpu_blelloch_scan(int* h_out, int* h_in, int n) {
    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 1<<4; // change based on size of n
    int num_blocks = (n + block_size * 2 - 1) / (block_size * 2);
    int shared_mem_size = block_size * 2 * sizeof(int);

    blelloch_scan_kernel<<<num_blocks, block_size, shared_mem_size>>>(d_out, d_in, n);

    // Print block sums
    int* h_block_sums = new int[num_blocks];
    int* d_block_sums;
    cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(int));
    extract_block_sums<<<1, num_blocks>>>(d_block_sums, d_out, n, block_size * 2);
    
    cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Block sums: ");
    for (int i = 0; i < num_blocks; i++) {
        printf("%d ", h_block_sums[i]);
    }
    printf("\n");
    
    // Clear print block sum artifacts
    delete[] h_block_sums;
    cudaFree(d_block_sums);


    // Handle block sums
    if (num_blocks > 1) {
        int* d_block_sums;
        cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(int));
        
        // Extract last element of each block
        extract_block_sums<<<1, num_blocks>>>(d_block_sums, d_out, n, block_size * 2);
        
        // Scan block sums
        blelloch_scan_kernel<<<1, num_blocks, num_blocks * sizeof(int)>>>(d_block_sums, d_block_sums, num_blocks);
        
        // Add block sums back
        add_block_sums<<<num_blocks, block_size>>>(d_out, d_block_sums, n, block_size * 2);
        
        cudaFree(d_block_sums);
    }

    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
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