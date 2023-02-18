#include <iostream>
#include <math.h>

__global__ void dot_product(float* vec1, float* vec2, float* result, int n) {
    __shared__ float shared_result[1024];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float temp_result = 0.0f;
    while (tid < n) {
        temp_result += vec1[tid] * vec2[tid];
        tid += blockDim.x * gridDim.x;
    }
    shared_result[threadIdx.x] = temp_result;

    __syncthreads();
    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            shared_result[threadIdx.x] += shared_result[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0) {
        result[blockIdx.x] = shared_result[0];
    }
}

int main() {
    int n = 1000000;

    // Allocate device memory
    float *d_vec1, *d_vec2, *d_result;
    cudaMalloc(&d_vec1, n * sizeof(float));
    cudaMalloc(&d_vec2, n * sizeof(float));
    cudaMalloc(&d_result, 1024 * sizeof(float)); // 分配设备端结果存储内存

    // Initialize host memory
    float *h_vec1 = new float[n];
    float *h_vec2 = new float[n];
    for (int i = 0; i < n; i++) {
        h_vec1[i] = sin(i);
        h_vec2[i] = cos(i);
    }

    // Copy host memory to device memory
    cudaMemcpy(d_vec1, h_vec1, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel on the device
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    dot_product<<<num_blocks, block_size>>>(d_vec1, d_vec2, d_result, n);

    // Copy result from device to host memory
    float *h_result = new float[num_blocks];
    cudaMemcpy(h_result, d_result, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute final result on host
    float final_result = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        final_result += h_result[i];
    }

    std::cout << "Dot product of two vectors: " << final_result << std::endl;

    // Free memory
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_result); // 释放设备端结果存储内存
    delete[] h_vec1;
    delete[] h_vec2;
    delete[] h_result;

    return 0;
}

