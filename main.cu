#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "scan.h"

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main() {
    const int n = 1<<7;
    int h_in[n], h_out_cpu_naive[n], h_out_cpu_blelloch[n], h_out_gpu_blelloch[n];

    // Initialize input array with random digits (0-9)
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        h_in[i] = rand() % 10;
    }

    // Print input array (first 10 elements)
    printf("Input: ");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%d ", h_in[i]);
    }
    printf("...\n");

    double start, end;

    // GPU Blelloch Scan
    start = get_time();
    gpu_blelloch_scan(h_out_gpu_blelloch, h_in, n);
    end = get_time();
    printf("GPU Blelloch Scan Time: %.6f seconds\n", end - start);

    // CPU Naive Sum --- doesn't work on non n = 2^k
    start = get_time();
    cpu_naive_sum(h_out_cpu_naive, h_in, n);
    end = get_time();
    printf("CPU Naive Sum Time: %.6f seconds\n", end - start);

    // CPU Blelloch Scan --- NOTE: this modifies the input array (also doesnt work on non n = 2^k)
    start = get_time();
    cpu_blelloch_scan(h_out_cpu_blelloch, h_in, n);
    end = get_time();
    printf("CPU Blelloch Scan Time: %.6f seconds\n", end - start);

    // Compare results
    bool cpu_naive_correct = true;
    bool cpu_blelloch_correct = true;
    bool gpu_blelloch_correct = true;

    for (int i = 0; i < n; i++) {
        if (h_out_cpu_naive[i] != h_out_gpu_blelloch[i]) {
            cpu_naive_correct = false;
        }
        if (h_out_cpu_blelloch[i] != h_out_gpu_blelloch[i]) {
            cpu_blelloch_correct = false;
        }
    }

    printf("CPU Naive Sum matches GPU Blelloch Scan: %s\n", cpu_naive_correct ? "Yes" : "No");
    printf("CPU Blelloch Scan matches GPU Blelloch Scan: %s\n", cpu_blelloch_correct ? "Yes" : "No");

    // Print output array (first 10 elements)
    printf("GPU Blelloch Scan Output: ");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%d ", h_out_gpu_blelloch[i]);
    }
    printf("...\n");

    // // Print output array (first 10 elements)
    // printf("CPU Blelloch Scan Output: ");
    // for (int i = 0; i < 10 && i < n; i++) {
    //     printf("%d ", h_out_cpu_blelloch[i]);
    // }
    // printf("...\n");

    // // Print output array (first 10 elements)
    // printf("CPU Naive Scan Output: ");
    // for (int i = 0; i < 10 && i < n; i++) {
    //     printf("%d ", h_out_cpu_naive[i]);
    // }
    // printf("...\n");

    return 0;
}