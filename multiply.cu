#include <stdio.h>

__global__ void multiply(int *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= 2;
    } else {
        printf("wrong idx %d", i);
    }
}

int main() {
    int n = 10;
    int a[n], *d_a;
    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
    printf("before mul\n");
     for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    cudaMalloc((void **)&d_a, n * sizeof(int));
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    multiply<<<2, 5>>>(d_a, n);
    cudaMemcpy(a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nafter mul\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
    cudaFree(d_a);
    return 0;
}

