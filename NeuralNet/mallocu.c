#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

static __host__ __device__ void *mallocu(int size)
{
	void* ptr;
#ifndef __CUDA_ARCH__
	ptr = malloc(size);
	if (ptr == NULL)
		fprintf(stderr, "Malloc error: Insufficient memory available");
#else
	cudaError_t result = cudaMalloc(&ptr, size);
	if (result != cudaSuccess)
		printf("CudaMalloc (size: %ld): (%d) %s\n", size, result, cudaGetErrorString(result)),
		ptr = NULL;
#endif
	return ptr;
}

static __host__ __device__ void freeu(void* ptr)
{
#ifndef __CUDA_ARCH__
	free(ptr);
#else
	cudaError_t result = cudaFree(ptr);
	printf("CudaFree (ptr: %p)\n", ptr);
	if (result != cudaSuccess)
		printf("CudaFree (ptr: %p): (%d) %s\n", ptr, result, cudaGetErrorString(result));
#endif
}
