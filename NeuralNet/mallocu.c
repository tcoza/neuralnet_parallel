#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
		fprintf(stderr, "CudaMalloc error: %d\n", result);
#endif
	return ptr;
}

static __host__ __device__ void* memcpyu(void* dst, void* src, size_t size) {
	void* ptr;
#ifndef __CUDA_ARCH__
	ptr = memcpy(dst, src, size);
	if (ptr == NULL)
		fprintf(stderr, "Memcpy error");
#else
	int result = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
	if (result != 0)
		fprintf(stderr, "CudaMemcpy error: %d\n", result);
	ptr = dst;
#endif
	return ptr;
}

static __host__ __device__ void freeu(void* ptr) {
#ifndef __CUDA_ARCH__
	free(ptr);
#else
	cudaFree(ptr);
#endif
}
