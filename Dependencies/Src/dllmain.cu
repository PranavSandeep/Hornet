#include "pch.h"
#include <Windows.h>
#include <string>
#include <cmath>

__global__ void cudaAddVectors(float* a, float* b, float* result, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void cudaMultiplyVectors(float* a, float* b, float* result, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        result[idx] = a[idx] * b[idx];
    }
}

extern "C" void addVectors(float* a, float* b, float* result, int size) {
    // Allocate device memory
    float* d_a, * d_b, * d_result;
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_b, size * sizeof(float));
    cudaMalloc((void**)&d_result, size * sizeof(float));

    // Copy input vectors to device
    cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaAddVectors << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_result, size);

    // Copy result back to host
    cudaMemcpy(result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

extern "C" float dot(float* a, float* b, float* result, int size)
{
    float* d_a, * d_b, * d_result;
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_b, size * sizeof(float));
    cudaMalloc((void**)&d_result, size * sizeof(float));

    //Copy input vectors to device
    cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaMultiplyVectors << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_result, size);

    // Copy result back to host
    cudaMemcpy(result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    float s = 0.f;

    for (int i = 0; i < size; i++)
    {
        s += result[i];
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return s;
}

extern "C" float magnitude(float* a, float* result, const int size)
{
    float s = 0.f;
    s = dot(a, a, result, size);

    return sqrt(s);
}




extern "C"
{

	__declspec(dllexport) int add(int a, int b)
	{
		return a + b;
	}

	__declspec(dllexport) void add_vec(float* a, float* b, float* result, int size)
	{
		addVectors(a, b, result, size);
	}

    __declspec(dllexport) float cdot(float* a, float* b,  float* result, int size)
    {
        float d = 0.f;
        d = dot(a, b, result, size);

        return d;
    }

    __declspec(dllexport) float cmagnitude(float* a, float* result, int size)
    {
        float m = 0.f;
        m = magnitude(a, result, size);

        return m;
    }
};