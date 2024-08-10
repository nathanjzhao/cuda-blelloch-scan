
#include <stdio.h>
#include "scan.h"

int main() {
    const int n = 1 << 8;
    int h_in[n], h_out[n];

    // Initialize input array
    for (int i = 0; i < n; i++) {
        h_in[i] = 1;
    }

    // Print input array (first 10 elements)
    printf("Input: ");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%d ", h_in[i]);
    }
    printf("...\n");


    gpu_blelloch_scan(h_out, h_in, n);

    // Print output array (first 10 elements)

    printf("Output: ");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%d ", h_out[i]);
    }
    printf("...\n");

}