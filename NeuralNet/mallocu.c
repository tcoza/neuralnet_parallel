#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

static __host__ __device__ void *mallocu(size_t size)
{
	void* ptr;
#ifndef __CUDA_ARCH__
	ptr = malloc(size);
	if (ptr == NULL)
		fprintf(stderr, "Malloc error: Insufficient memory available");
#else
	int result = cudaMalloc(&ptr, size);
	if (result != 0)
		printf("CudaMalloc error: %ld (requested size: %ld)\n", result, size),
		ptr = NULL;
#endif
	return ptr;
}

static __host__ __device__ void freeu(void* ptr)
{
#ifndef __CUDA_ARCH__
	free(ptr);
#else
	cudaFree(ptr);
#endif
}
