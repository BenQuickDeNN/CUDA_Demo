#include "cuda_vector.h"

#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//void vector_add();

/**
* @brief 核函数-向量加
* @param x 输入向量
* @param y 输入向量
* @param z 输出向量
* @param n 向量长度
*/
__global__ void cuda_vector_add(float *x, float *y, float *z, int n)
{
	// 获取全局索引（向量数组的索引）
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// 获取步长（一次批处理的长度）
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) z[i] = x[i] + y[i];
}

/**
* @brief Host端函数-向量加
*/
void vector_add()
{
	printf("----------------计算向量加----------------\r\n");
	// 向量长度
	int n = 1 << 24;
	int nBytes = n * sizeof(float);
	// 申请host内存
	float *x, *y, *z;
	x = (float*)malloc(nBytes);
	y = (float*)malloc(nBytes);
	z = (float*)malloc(nBytes);
	// 数据初始化
	for (int i = 0; i < n; i++)
	{
		x[i] = 10.0;
		y[i] = 20.0;
	}
	// 申请device内存
	float *d_x, *d_y, *d_z;
	cudaMalloc((void**)&d_x, nBytes);
	cudaMalloc((void**)&d_y, nBytes);
	cudaMalloc((void**)&d_z, nBytes);
	// 将数据从host拷贝到device
	cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
	// 定义kernel
	dim3 blockSize(256); // GTX 710 block中线程个数不能超过1024
	dim3 gridSize((n + blockSize.x - 1) / blockSize.x); // gridSize个block可容纳整个向量
	printf("blockSize = %d\r\n", blockSize.x);
	printf("gridSize = %d\r\n", gridSize.x);
	// 执行计算
	cuda_vector_add << < gridSize, blockSize >> > (d_x, d_y, d_z, n);
	// 将数据从device拷贝到host
	cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);
	// 检查结果
	float maxError = 0.0;
	for (int i = 0; i < n; i++) maxError += (z[i] - 30.0) * (z[i] - 30.0);
	printf("误差平方和: %f\r\n", maxError);
	// 释放device内存
	cudaFree(d_z);
	cudaFree(d_y);
	cudaFree(d_x);
	// 释放host内存
	free(z);
	free(y);
	free(x);
}