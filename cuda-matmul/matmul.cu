#include <cstdio>
#include <cuda_runtime.h>

__global__ void matmulKernel(const float* M, const float* N, float* P, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= width || col >= width) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < width; ++k) {
        sum += M[row * width + k] * N[k * width + col];
    }
    P[row * width + col] = sum;
}

int main()
{
    constexpr int width = 20;
    constexpr int elements = width * width;
    float M[elements];
    float N[elements];
    float P[elements];

    for (int i = 0; i < elements; ++i) {
        M[i] = static_cast<float>(i + 1);
        N[i] = (i / width == i % width) ? 1.0f : 0.0f;
    }

    const auto bytes = width * width * sizeof(float);
    float* deviceM;
    float* deviceN;
    float* deviceP;

    // Allocate GPU memory and copy the input matrices.
    cudaMalloc(&deviceM, bytes);
    cudaMalloc(&deviceN, bytes);
    cudaMalloc(&deviceP, bytes);
    cudaMemcpy(deviceM, M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceN, N, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel. Each valid thread computes one output element.
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (width + block.y - 1) / block.y);
    matmulKernel<<<grid, block>>>(deviceM, deviceN, deviceP, width);

    // Copy the result back and release GPU memory.
    cudaMemcpy(P, deviceP, bytes, cudaMemcpyDeviceToHost);
    cudaFree(deviceM);
    cudaFree(deviceN);
    cudaFree(deviceP);

    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            std::printf("%5.0f", P[row * width + col]);
        }
        std::printf("\n");
    }
}
