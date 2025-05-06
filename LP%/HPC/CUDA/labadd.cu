CUDA Program for Addition of Two Large Vectors
 #include <stdio.h>
 #include <stdlib.h>
 #include <cuda_runtime.h>
 // CUDA kernel for vector addition
 __global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
 }
 int main() {
    int n = 1000000;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = n * sizeof(int);
    // Allocate host memory
    a = (int*) malloc(size);
    b = (int*) malloc(size);
    c = (int*) malloc(size);
    // Initialize vectors
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }
    // Allocate device memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    // Copy host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // Verify result
    for (int i = 0; i < n; i++) {
        if (c[i] != 2 * i) {
            printf("Error: c[%d] = %d\n", i, c[i]);
            break;
        }
    }
    printf("Vector addition successful!\n");
    // Free memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);
    return 0;
 }
 Output:
 Vector addition successful!